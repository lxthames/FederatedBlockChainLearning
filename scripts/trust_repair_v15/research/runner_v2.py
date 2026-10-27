"""Disjoint evaluation, independent randomness and auditable recovery protocol."""
import copy
import hashlib
import json
import math
import platform
import time
import warnings
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .runner_helpers import _attack, _defense, _apply_delta, _state_delta, _update_bytes
from .attacks import AttackContext, LowAndSlowAttack
from .contracts import ClientUpdate, AggregationResult, VoteDecision
from .defenses import TwoLayerDefense, AggregationContext, flatten_update, MultiKrumDefense
from .committee import CommitteePolicy
from .validation import LocalValidator, VoteAggregator
from .identity import IdentityOracle
from .ledger import AppendOnlyRecoveryLedger
from .accountability import AccountabilityManager
from .detector import TrendDetector, AdjacentDetector, DetectionResult
from ..fedval import FedVal
from ..adaptive import constrained_replacement
from ..loss_voting import LossCommittee, clip_updates
from ..repair import RepairRound as ChallengeRound
from ..recovery_guard import checkpoint_benefit
from ..literature import LiteratureDefense
from .events import EventLogger
from .splits import stream, make_splits, fingerprint, matched_test_shards, freeze_evaluation_and_redistribute
from ..pairing import split_bindings, digest_value, TRAINING_FIELDS


def model_hash(model):
    return fingerprint(*[x.detach().cpu().numpy() for x in model.state_dict().values()])


class BenchmarkRunner:
    def __init__(self, config, output_dir, *, mass_schedule=None, mass_source_binding=None):
        self.challenge_arm = 'current'
        self.challenge_thresholds = None
        self.config = config
        if config.aggregation_mass_policy=='replay':
            if mass_schedule is None or len(mass_schedule)!=config.rounds or not mass_source_binding:
                raise ValueError('replay requires a complete bound mass schedule')
            if any(not math.isfinite(v) or not 0<=v<=1 for v in mass_schedule):
                raise ValueError('invalid replay mass schedule')
        elif mass_schedule is not None or mass_source_binding is not None:
            raise ValueError('mass schedule only allowed for replay')
        self.mass_schedule = list(mass_schedule) if mass_schedule is not None else None
        self.diagnostic_input_binding = {'mass_schedule':self.mass_schedule,'source':mass_source_binding}
        self.local_training_work = {'optimizer_steps':0,'processed_examples':0}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if (self.output_dir / 'events.jsonl').exists():
            raise FileExistsError('refusing to overwrite an existing run')
        self.logger = EventLogger(self.output_dir / 'events.jsonl', config.experiment_id, config.seed)
        self.device = torch.device('cuda' if config.device == 'auto' and torch.cuda.is_available()
                                   else ('cpu' if config.device == 'auto' else config.device))
        self.trace = []
        self.validation_work = {'private_example_evaluations':0,'private_forward_passes':0,'reference_example_evaluations':0,'reference_forward_passes':0}

    def _score(self, model, features, labels):
        model.eval()
        total = 0.0
        with torch.no_grad():
            for start in range(0, len(features), self.config.evaluation_batch_size):
                x = torch.from_numpy(np.asarray(features[start:start+self.config.evaluation_batch_size], dtype=np.float32)).to(self.device)
                y = torch.as_tensor(labels[start:start+len(x)], device=self.device)
                output = model(x)
                if y.ndim == 1:
                    total += (output.argmax(1) == y).sum().item()
                else:
                    selected = output.topk(min(self.config.routing_budget, output.shape[1]), dim=1).indices
                    total += (y.gather(1, selected).sum(1) / y.sum(1).clamp_min(1e-12)).sum().item()
        return total / max(1, len(features))

    def _apply_trigger(self, features):
        triggered = features.copy()
        if triggered.ndim == 4:
            triggered[:, :, -2:, -2:] = 0.0 if self.config.dataset == 'femnist' else 1.0
        else:
            triggered[..., -1] += 5.0
        return triggered

    def _train(self, model, x, y, round_index, client, poison=False):
        c = self.config
        x, y = x.copy(), y.copy()
        if c.benign_drift_round and round_index >= c.benign_drift_round:
            x += c.benign_drift_magnitude
        if poison and y.ndim == 1:
            selected = stream(c.seed, 'backdoor', round_index, client).permutation(len(y))[:max(1, math.ceil(len(y)*c.backdoor_fraction))]
            x[selected] = self._apply_trigger(x[selected])
            y[selected] = c.backdoor_target_label
        # Model dropout and shuffling are independent of defense-specific work.
        seed = int(stream(c.seed, 'training', round_index, client+100).integers(2**31))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                            batch_size=c.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(seed))
        optimizer = torch.optim.SGD(model.parameters(), lr=c.learning_rate, momentum=c.optimizer_momentum)
        model.train()
        max_steps=c.local_steps if c.local_training_budget=='fixed_steps' else None
        local_step_count=0
        # Restart the deterministic shuffled loader only if an exact fixed-step
        # budget exceeds one pass. Epoch mode retains V11's original sequence.
        passes=math.ceil(max_steps/max(1,len(loader))) if max_steps else c.local_epochs
        for _ in range(passes):
            for bx, by in loader:
                if max_steps is not None and local_step_count>=max_steps:break
                # BatchNorm cannot estimate variance from a singleton batch.
                if len(bx) == 1 and any(isinstance(m, nn.BatchNorm1d) for m in model.modules()):
                    model.eval()
                else:
                    model.train()
                optimizer.zero_grad()
                output = model(bx.to(self.device))
                by = by.to(self.device)
                if by.ndim == 1:
                    loss = nn.functional.cross_entropy(output, by.long())
                else:
                    target = by / by.sum(1, keepdim=True).clamp_min(1e-12)
                    probabilities = output if c.model == 'emh_dnn' else output.softmax(1)
                    loss = nn.functional.mse_loss(probabilities, target)
                loss.backward()
                optimizer.step()
                local_step_count += 1
                self.local_training_work['optimizer_steps'] += 1
                self.local_training_work['processed_examples'] += len(bx)

    def run_arrays(self, features, labels, model_factory, test_arrays=None):
        c = self.config
        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(labels, dtype=np.int64 if labels.ndim == 1 else np.float32)
        if len(x) != len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('incompatible or non-finite input arrays')
        if c.attack in {'model_replacement','adaptive_model_replacement'} and y.ndim != 1:
            raise ValueError('targeted model replacement requires classification labels')
        if y.ndim == 1 and not 0 <= c.backdoor_target_label < int(y.max())+1:
            raise ValueError('target label outside task')
        split = make_splits(y, c, external_test=test_arrays is not None)
        tx, ty = (x[split.test], y[split.test]) if test_arrays is None else test_arrays
        tx = np.asarray(tx, dtype=np.float32)
        ty = np.asarray(ty, dtype=y.dtype)
        split = freeze_evaluation_and_redistribute(split, y, ty, c)
        bindings = split_bindings(split)
        classes = y if y.ndim == 1 else y.argmax(1)
        torch.manual_seed(c.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
        model = model_factory().to(self.device)
        initial_model_hash = model_hash(model)
        self.logger.log('environment', 0, {'python_version': platform.python_version(),
            'torch_version': torch.__version__, 'numpy_version': np.__version__,
            'device': str(self.device), 'gpu': torch.cuda.get_device_name(0) if self.device.type == 'cuda' else None,
            'config': c.to_dict(), 'data_fingerprint': fingerprint(x, y),
            'test_fingerprint': fingerprint(tx, ty)})
        source_paths=sorted(Path(__file__).resolve().parents[1].rglob('*.py'))
        source_map={str(p.relative_to(Path(__file__).resolve().parents[1])):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
        source_hash=hashlib.sha256(json.dumps(source_map,sort_keys=True).encode()).hexdigest()
        (self.output_dir / 'config.json').write_text(json.dumps(c.to_dict(), indent=2), encoding='utf-8')
        manifest = split.manifest()
        manifest['external_test'] = test_arrays is not None
        (self.output_dir / 'splits.json').write_text(json.dumps(manifest), encoding='utf-8')
        malicious = set(stream(c.seed, 'poisoning-identities').choice(c.num_clients, math.floor(c.num_clients*c.malicious_fraction), replace=False).tolist())
        corrupt = set(stream(c.seed, 'validator-identities').choice(c.num_clients, math.floor(c.num_clients*c.malicious_validator_fraction), replace=False).tolist())
        duplicates = set(stream(c.seed, 'duplicate-identities').choice(c.num_clients, math.floor(c.num_clients*c.duplicate_identity_fraction), replace=False).tolist())
        oracle = IdentityOracle(c.identity_false_accept_rate, c.identity_false_reject_rate)
        enrollments = [oracle.enroll(i, i in duplicates, stream(c.seed, 'enrollment', i)) for i in range(c.num_clients)]
        enrolled = [d.client_id for d in enrollments if d.accepted]
        if not enrolled:
            raise ValueError('identity enrollment left no eligible clients')
        self.logger.log('enrollment', 0, {'decisions': [asdict(d) for d in enrollments]})
        ledger = AppendOnlyRecoveryLedger()
        ledger.checkpoint(0, {k:v.detach().cpu() for k,v in model.state_dict().items()})
        ledger.append_model_commit(0, {'round':0, 'model_hash':model_hash(model)})
        self.trace.append({'kind':'genesis', 'model_hash':model_hash(model), 'enrolled':enrolled})
        chain = None
        if c.ledger_backend == 'evm':
            from .evm import EVMAdapter
            chain = EVMAdapter(c, self.output_dir)
            chain.genesis(self.trace[0])
        accountability = AccountabilityManager(c.slashing_penalty)
        accountability.initialize({i:c.initial_stake for i in enrolled})
        layered = c.defense == 'two_layer'
        defense = TwoLayerDefense(c.initial_trust,
            c.trust_reward if c.trust_updates == 'dynamic' else 0.,
            c.rejection_trust_penalty if c.trust_updates == 'dynamic' and c.rejection_trust_policy == 'rank_rejection' else 0.,
            c.probation_weight_floor, c.trust_weighting, c.legacy_inconclusive_penalty) if layered else None
        detector_cls = TrendDetector if c.detector == 'trend' else AdjacentDetector
        detector = detector_cls(c.detector_window, c.relative_drop_threshold, c.slope_threshold)
        fedval = FedVal(self, x[split.reference], y[split.reference], int(y.max())+1) if c.defense == 'fedval' else None
        literature = LiteratureDefense(self,x,y,split) if c.defense in {'flame','flshield_cluster','fldetector_pp'} else None
        validator_monitors = {}
        intervention_signers = set()
        monitoring_quorum_failures = 0
        rejection_counts = {i: 0 for i in enrolled}
        submission_counts = {i: 0 for i in enrolled}
        eligibility_counts = {i: 0 for i in enrolled}
        history, before_history, asrs, before_asrs, times = [], [], [], [], []
        clean_trigger_history=[]
        submitted = accepted_bad = recoveries = wrongful = transmitted = unsupported = 0
        tp = fp = fn = tn = 0
        panel, window_rounds, window_signers = [], [], set()
        panel_start = 0
        active_version = 0
        commits = []
        monitor_values = []
        first_attack = delay = None
        participant_history = []
        inconclusive_counts = {i: 0 for i in enrolled}
        quality_rejection_counts = {i: 0 for i in enrolled}
        unavailable_counts = {i: 0 for i in enrolled}
        rank_vote_counts = {'rank_rejected_locally_helpful_or_neutral': 0,
                            'rank_approved_locally_harmful': 0, 'total': 0}
        for rnd in range(1, c.rounds+1):
            started = time.perf_counter()
            count=min(c.participants_per_round,len(enrolled))
            participant_rng=stream(c.seed,'participants',rnd)
            if c.sampling_policy=='bounded':
                # A controlled Byzantine-budget experiment: sampling uses labels
                # only to enforce the attack schedule, never to filter updates.
                bad_pool=[i for i in enrolled if i in malicious]
                good_pool=[i for i in enrolled if i not in malicious]
                bad_budget=min(math.floor(count*c.malicious_fraction),len(bad_pool))
                good_count=count-bad_budget
                if good_count>len(good_pool):
                    raise ValueError('enrollment left too few honest clients for the declared participation budget')
                participants=sorted(participant_rng.choice(bad_pool,bad_budget,replace=False).tolist()+
                    participant_rng.choice(good_pool,good_count,replace=False).tolist())
            else:
                participants=sorted(participant_rng.choice(enrolled,count,replace=False).tolist())
            participant_history.append(participants)
            transmitted += sum(v.numel()*v.element_size() for v in model.state_dict().values())*len(participants)
            eligible = [i for i in enrolled if (not layered or
                (accountability.balance(i) >= c.minimum_stake and (defense.trust_score(i) if c.trust_eligibility == 'dynamic' else c.initial_trust) >= c.minimum_trust))]
            for i in eligible:
                eligibility_counts[i] += 1
            updates = []
            attack_active = rnd >= c.attack_start_round and c.attack != 'none'
            for client in participants:
                # Low trust prevents validation but permits reviewed probationary
                # proposals. Depleted stake still prevents subsequent participation.
                if layered and accountability.balance(client) < c.minimum_stake:
                    continue
                indices = split.train[client]
                local = copy.deepcopy(model)
                work_before=dict(self.local_training_work)
                self._train(local, x[indices], y[indices], rnd, client,
                            poison=attack_active and client in malicious and c.attack in {'model_replacement','adaptive_model_replacement'})
                delta = _state_delta(local, model)
                self.logger.log('local_update_diagnostic',rnd,{'client_id':client,'sample_count':len(indices),
                    'optimizer_steps':self.local_training_work['optimizer_steps']-work_before['optimizer_steps'],
                    'processed_examples':self.local_training_work['processed_examples']-work_before['processed_examples'],
                    'unclipped_norm':float(flatten_update(ClientUpdate(client,delta,len(indices))).norm())})
                metadata = {}
                if attack_active and client in malicious and c.attack == 'adaptive_model_replacement':
                    honest_local = copy.deepcopy(model)
                    self._train(honest_local, x[indices], y[indices], rnd, client)
                    delta, metadata = constrained_replacement(self, model, _state_delta(honest_local, model), delta, x[indices], y[indices])
                updates.append(ClientUpdate(client, delta, len(indices), client in malicious, metadata))
                submission_counts[client] += 1
            bad_count = sum(u.is_malicious for u in updates) if attack_active else 0
            active = bad_count > 0
            if active and first_attack is None:
                first_attack = rnd
            attack = LowAndSlowAttack(c.gradual_max_norm_ratio,c.gradual_ramp_rounds) if c.attack == 'low_and_slow' else _attack('none' if c.attack == 'adaptive_model_replacement' else c.attack)
            honest = tuple(u for u in updates if not u.is_malicious)
            attacked = []
            for u in updates:
                if u.is_malicious and attack_active:
                    attack_round = rnd-c.attack_start_round+1
                    u = attack.apply(u, AttackContext(attack_round,c.seed,len(updates),max(1,bad_count),honest))
                attacked.append(u)
            submitted += bad_count
            if c.archive_updates:
                archive=self.output_dir/'offchain-history'
                archive.mkdir(exist_ok=True)
                torch.save({'round':rnd,'model_state':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                    'updates':[{'client_id':u.client_id,'state':dict(u.state),'sample_count':u.sample_count} for u in attacked]},
                    archive/f'round-{rnd:03}.pt')
            if layered:
                attacked,clipping=clip_updates(attacked,c.candidate_clip_ratio)
                self.logger.log('candidate_clipping',rnd,clipping)
            transmitted += sum(_update_bytes(u) for u in attacked)
            self.challenge = ChallengeRound(self, model, x, y, split, rnd, attacked, eligible)
            decisions, candidate_trace = {}, []
            if layered:
                loss_committee=LossCommittee(self,model,x,y,split,attacked) if c.vote_rule!='accuracy' else None
                for u in attacked:
                    pool = [i for i in eligible if i != u.client_id]
                    policy = CommitteePolicy().for_population(len(pool))
                    if not policy.consensus_supported:
                        unavailable_counts[u.client_id] += 1
                        self.logger.log('committee_unavailable', rnd, {'client_id': u.client_id, 'eligible_snapshot': pool})
                        continue
                    members = sorted(stream(c.seed,'committee',rnd,u.client_id).choice(pool,policy.size,replace=False).tolist())
                    votes = []
                    candidate = copy.deepcopy(model)
                    _apply_delta(candidate,dict(u.state))
                    for validator in members:
                        if loss_committee is not None:
                            if validator in corrupt and c.validator_behavior=='withhold':continue
                            forced=(u.client_id in malicious) if validator in corrupt and c.validator_behavior=='approve_bad_reject_good' else None
                            vote,diagnostic=loss_committee.vote(validator,u.client_id,validator in corrupt,forced)
                            if diagnostic.get('abstain'):
                                self.logger.log('loss_vote_abstention',rnd,dict(diagnostic,validator=validator,candidate=u.client_id))
                                continue
                            votes.append(vote)
                            if c.vote_rule=='rank_loss' and 'locally_harmful' in diagnostic:
                                rank_vote_counts['total'] += 1
                                rank_vote_counts['rank_rejected_locally_helpful_or_neutral'] += int(
                                    diagnostic['relative_rank_rejected'] and not diagnostic['locally_harmful'])
                                rank_vote_counts['rank_approved_locally_harmful'] += int(
                                    not diagnostic['relative_rank_rejected'] and diagnostic['locally_harmful'])
                            self.logger.log('loss_vote',rnd,dict(diagnostic,validator=validator,candidate=u.client_id,approve=vote.approve))
                            continue
                        vi = split.validation[validator]
                        baseline = self._score(model,x[vi],y[vi])
                        score = self._score(candidate,x[vi],y[vi])
                        if validator in corrupt and c.validator_behavior == 'withhold':
                            continue
                        if validator in corrupt and c.validator_behavior == 'approve_bad_reject_good':
                            score = baseline if u.client_id in malicious else baseline-1.0
                        vote = LocalValidator(validator,c.minimum_validation_delta,c.validation_absolute_tolerance).vote(u.client_id,baseline,score,validator in corrupt)
                        votes.append(vote)
                    decisions[u.client_id] = VoteAggregator().decide(votes, policy.quorum, members)
                    decision = decisions[u.client_id]
                    inconclusive_counts[u.client_id] += int(decision.outcome == 'insufficient_response')
                    quality_rejection_counts[u.client_id] += int(decision.outcome == 'quality_rejected')
                    delta_hash = fingerprint(*[v.numpy() for v in u.state.values()])
                    candidate_trace.append({'client_id':u.client_id,'candidate_hash':delta_hash,
                        'eligible_snapshot':pool,
                        'committee':members, 'quorum':policy.quorum,
                        'votes':[{'validator_id':v.validator_id,'approve':v.approve} for v in votes],
                        'accepted':decisions[u.client_id].accepted, 'decision':asdict(decisions[u.client_id])})
                    transmitted += _update_bytes(u)*len(members) + len(members)*96
                if c.vote_rule=='rank_loss':
                    # Ranking requires comparator updates even outside a candidate's
                    # own committee. Count those additional model deliveries.
                    assigned={r['client_id']:set(r['committee']) for r in candidate_trace}
                    union=set().union(*assigned.values()) if assigned else set()
                    transmitted += sum(_update_bytes(u)*len(union-assigned.get(u.client_id,set())-{u.client_id}) for u in attacked)
                supported = [u for u in attacked if u.client_id in decisions]
                if supported:
                    result = defense.aggregate(model.state_dict(),supported,AggregationContext(rnd,decisions))
                    if c.vote_rule=='scaled_loss' and c.aggregation_mass_policy=='nominal':
                        # Keep nominal sample shares; rejected mass becomes zero
                        # contribution instead of amplifying survivors after voting.
                        accepted_mass=sum(u.sample_count for u in attacked if u.client_id in result.accepted_client_ids)
                        total_mass=sum(u.sample_count for u in attacked)
                        fraction=accepted_mass/max(1,total_mass)
                        result=AggregationResult({k:v*fraction if v.is_floating_point() else v for k,v in result.state.items()},
                            result.accepted_client_ids,result.rejected_client_ids,result.diagnostics)
                    if c.aggregation_mass_policy=='replay':
                        factor=self.mass_schedule[rnd-1]
                        result=AggregationResult({k:v*factor if v.is_floating_point() else v for k,v in result.state.items()},
                            result.accepted_client_ids,result.rejected_client_ids,result.diagnostics)
                        self.logger.log('diagnostic_mass_replay',rnd,{'factor':factor,'source':self.diagnostic_input_binding['source']})
                    if not result.accepted_client_ids:
                        unsupported += 1
                else:
                    unsupported += 1
                    result = AggregationResult({k:torch.zeros_like(v).cpu() for k,v in model.state_dict().items()},())
            else:
                byzantine_bound = math.ceil(len(attacked)*c.malicious_fraction) if c.attack != 'none' else 0
                if c.defense in {'multi_krum','boba'} and bad_count>byzantine_bound:
                    raise ValueError('sampled attackers exceed the declared Byzantine budget')
                trusted = server_vectors = None
                if c.defense in {'fltrust','boba'}:
                    root = split.reference
                    if not len(root):
                        raise ValueError('trusted reference pool is empty')
                    reference_model = copy.deepcopy(model)
                    self._train(reference_model,x[root],y[root],rnd,-1)
                    trusted = _state_delta(reference_model,model)
                    if c.defense == 'boba':
                        vectors = []
                        for cls in np.unique(classes[root]):
                            indices = root[classes[root] == cls]
                            ref = copy.deepcopy(model)
                            self._train(ref,x[indices],y[indices],rnd,-2-int(cls))
                            vectors.append(flatten_update(ClientUpdate(-1,_state_delta(ref,model),len(indices))))
                        server_vectors = torch.stack(vectors)
                if literature:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        result = literature.aggregate(model,attacked,rnd)
                    self.logger.log('literature_defense',rnd,dict(result.diagnostics,
                        algorithm_warnings=[str(w.message) for w in caught]))
                    transmitted += int(result.diagnostics.get('extra_transmitted_bytes',0))
                elif fedval:
                    result = fedval.aggregate(model, attacked)
                    self.logger.log('fedval_weights', rnd, dict(result.diagnostics))
                else:
                    baseline = _defense(c.defense,byzantine_bound,trusted,server_vectors)
                    if c.defense == 'multi_krum':
                        baseline = MultiKrumDefense(byzantine_bound,max(1,len(attacked)-2*byzantine_bound-2))
                    result = baseline.aggregate(model.state_dict(),attacked,None)
            result, challenge_blocked = self.challenge.aggregate(result)
            transmitted += self.challenge.extra_transmitted_bytes
            unsupported += int(challenge_blocked)
            _apply_delta(model,dict(result.state))
            before = self._score(model,tx,ty)
            before_history.append(before)
            targeted = c.attack in {'model_replacement','adaptive_model_replacement'} and y.ndim == 1
            trigger_control = c.attack == 'none' and y.ndim == 1
            target_mask = ty != c.backdoor_target_label if targeted or trigger_control else None
            def asr():
                if not (targeted or trigger_control) or not np.any(target_mask):
                    return None
                return self._score(model,self._apply_trigger(tx[target_mask]),np.full(int(target_mask.sum()),c.backdoor_target_label,dtype=np.int64))
            pre_asr = asr()
            before_asrs.append(pre_asr)
            signers = sorted({s for i in result.accepted_client_ids for s in decisions[i].affirmative_signers}) if layered else []
            intervention_signers.update(signers if rnd > c.intervention_checkpoint else [])
            for u in attacked:
                if u.client_id not in result.accepted_client_ids:
                    rejection_counts[u.client_id] += 1
            parent = active_version
            active_version = rnd
            commit = {'kind':'round','round':rnd,'parent_version':parent,'model_hash':model_hash(model),
                'candidates':candidate_trace,'accepted_client_ids':list(result.accepted_client_ids),
                'affirmative_signers':signers,'recovery':None}
            ledger.checkpoint(rnd,{k:v.detach().cpu() for k,v in model.state_dict().items()})
            ledger.append_model_commit(rnd,{k:v for k,v in commit.items() if k != 'candidates'})
            commits.append(commit)
            detection = None
            if layered and c.ablation != 'validation_only':
                if not panel or len(window_rounds) >= (c.detector_window-1 if c.detector == 'trend' else 1):
                    detector.reset()
                    window_rounds, window_signers = [], set()
                    panel_start = parent
                    policy = CommitteePolicy().for_population(len(eligible))
                    panel = sorted(stream(c.seed,'monitor-panel',rnd).choice(eligible,policy.size,replace=False).tolist()) if policy.consensus_supported else []
                    if panel:
                        previous = copy.deepcopy(model)
                        previous.load_state_dict(ledger._checkpoints[parent])
                        validator_monitors = {i: detector_cls(c.detector_window,c.relative_drop_threshold,c.slope_threshold) for i in panel}
                        initial = [self._score(previous,x[split.validation[i]],y[split.validation[i]]) for i in panel]
                        for i, score in zip(panel, initial):
                            validator_monitors[i].observe(score)
                        detector.observe(float(np.median(initial)))
                if panel:
                    window_rounds.append(rnd)
                    window_signers.update(signers)
                    reported = {}
                    attestations = []
                    for i in panel:
                        score = self._score(model,x[split.validation[i]],y[split.validation[i]])
                        if i in corrupt and c.validator_behavior == 'false_recovery':
                            score = max(0.0, 1.0-0.24*len(window_rounds))
                        elif i in corrupt and c.validator_behavior == 'approve_bad_reject_good':
                            score = 1.0
                        reported[i] = score
                        local_flag = validator_monitors[i].observe(score)
                        if local_flag.detected and not (i in corrupt and c.validator_behavior == 'withhold'):
                            attestations.append(i)
                    monitoring = float(np.median(list(reported.values())))
                    transmitted += len(panel)*96
                    monitor_values.append(monitoring)
                    detection = detector.observe(monitoring)
                    quorum = math.ceil(2*len(panel)/3)
                    if detection.detected and len(attestations) < quorum:
                        monitoring_quorum_failures += 1
                        detection = DetectionResult(False,'insufficient_recovery_attestations',detection.relative_drop,detection.slope)
                    if c.recovery_policy != 'protocol':
                        scheduled = c.recovery_policy.startswith('scheduled') and rnd == c.intervention_round
                        detection = DetectionResult(scheduled,'scheduled_intervention' if scheduled else None,0.0,0.0)
                        if scheduled:
                            panel_start = c.intervention_checkpoint
                            window_rounds = list(range(c.intervention_checkpoint+1,rnd+1))
                            window_signers = set(intervention_signers)
                            attestations = list(panel)
                    accuracy_detection = asdict(detection)
                    direct_loss = c.recovery_signal == 'paired_loss_quorum'
                    window_ready = len(window_rounds) == c.detector_window-1
                    check_benefit = window_ready if direct_loss else detection.detected
                    if direct_loss and c.recovery_policy == 'protocol':
                        detection = DetectionResult(False,'awaiting_loss_window',0.0,0.0)
                    if check_benefit and c.recovery_benefit_guard and c.recovery_policy=='protocol':
                        checkpoint=copy.deepcopy(model)
                        checkpoint.load_state_dict(ledger._checkpoints[panel_start])
                        evaluator=LossCommittee(self,model,x,y,split,[])
                        benefit_signers=[];benefit_reports={}
                        for validator in panel:
                            ids=split.validation[validator]
                            good,report=checkpoint_benefit(evaluator.losses(model,ids),evaluator.losses(checkpoint,ids),
                                c.recovery_loss_margin,c.recovery_standard_errors)
                            if validator in corrupt:
                                good=c.validator_behavior=='false_recovery'
                            benefit_reports[validator]=report
                            if good and not (validator in corrupt and c.validator_behavior=='withhold'):
                                benefit_signers.append(validator)
                        # Direct loss mode is one paired test per complete fixed window.
                        # It never consults attack labels, test examples or trigger inputs.
                        attestations=sorted(set(benefit_signers) if direct_loss else set(attestations)&set(benefit_signers))
                        self.logger.log('recovery_confirmation',rnd,{'checkpoint':panel_start,'panel':panel,
                            'reports':benefit_reports,'affirmative':attestations,'quorum':quorum,
                            'signal':c.recovery_signal,'accuracy_detection':accuracy_detection})
                        if direct_loss:
                            enough=len(attestations)>=quorum
                            detection=DetectionResult(enough,'paired_loss_quorum' if enough else 'checkpoint_benefit_not_confirmed',0.0,0.0)
                        if len(attestations)<quorum:
                            detection=DetectionResult(False,'checkpoint_benefit_not_confirmed',detection.relative_drop,detection.slope)
                    self.logger.log('monitor',rnd,{'panel':panel,'score':monitoring,'window_start_version':panel_start,
                        'reports':reported,'affirmative_recovery_attesters':attestations,'quorum':quorum,
                        'accuracy_detection':accuracy_detection,'final_detection':asdict(detection)})
                    if detection.detected:
                        penalties_enabled=c.recovery_penalties_enabled and c.ablation!='no_penalties' and c.recovery_policy!='scheduled_plain'
                        penalized = sorted(window_signers) if penalties_enabled else []
                        restored, record = ledger.recover(rnd,panel_start,detection.reason,tuple(penalized))
                        model.load_state_dict(restored)
                        active_version = panel_start
                        recoveries += 1
                        slashed = accountability.slash_affirmative_signers(tuple(penalized),corrupt)
                        wrongful += slashed.wrongful_slashing_count
                        for i in penalized:
                            defense.set_trust_score(i,max(0.0,defense.trust_score(i)-c.trust_penalty))
                        commit['recovery'] = {'from_round':window_rounds[0], 'to_round':rnd,
                            'restored_version':panel_start,'signers':penalized,'panel':panel,
                            'attesters':attestations,
                            'penalties_enabled':penalties_enabled,'trigger':detection.reason}
                        detector.reset()
                        panel=[]
                    # This is attack-presence detection, not proof that an attack caused damage.
                    if detection.detected and active:
                        tp += 1
                        if delay is None:
                            delay = rnd-first_attack
                    elif detection.detected: fp += 1
                    elif active: fn += 1
                    else: tn += 1
            self.trace.append(commit)
            if chain:
                chain.round(commit)
                for client in enrolled:
                    if chain.contract.functions.stake(chain.account(client)).call() != chain.units(accountability.balance(client)):
                        raise AssertionError('simulator/contract stake mismatch')
            accepted_bad += sum(i in malicious for i in result.accepted_client_ids) if attack_active else 0
            history.append(self._score(model,tx,ty))
            post_asr = asr()
            if post_asr is not None and targeted: asrs.append(post_asr)
            if post_asr is not None and trigger_control: clean_trigger_history.append(post_asr)
            times.append(time.perf_counter()-started)
            self.logger.log('round_completed',rnd,{'participants':participants,
                'accepted_client_ids':list(result.accepted_client_ids),
                'accuracy':history[-1],'utility_before_recovery':before,
                'targeted_attack_success':post_asr,'targeted_asr_before_recovery':pre_asr,
                'committee_decisions':{str(i):asdict(d) for i,d in decisions.items()},
                'validator_eligible_ids':eligible,
                'trust_after_round':{str(i):defense.trust_score(i) for i in enrolled} if layered else None,
                'detection':asdict(detection) if detection else None,
                'runtime_seconds':times[-1], 'active_version':active_version})
        # Independent held-out test shards for fairness; never private validation scores.
        test_partitions = matched_test_shards(y,ty,split,c.seed)
        utilities = [self._score(model,tx[i],ty[i]) for i in test_partitions if len(i)]
        group_config = {k:v for k,v in c.to_dict().items() if k not in {'experiment_id','seed','device','rpc_url','deployment_path','natural_splits_path'}}
        summary = {'configuration':c.to_dict(),'source_hash':source_hash,
            'initial_model_hash':initial_model_hash, 'split_bindings':bindings,
            'training_binding':digest_value({k:c.to_dict()[k] for k in TRAINING_FIELDS}),
            'participants':participant_history,
            'inconclusive_counts':inconclusive_counts,
            'quality_rejection_counts':quality_rejection_counts,
            'committee_unavailable_counts':unavailable_counts,
            'rank_vote_counts':rank_vote_counts,
            'validation_work':self.validation_work,
            'local_training_work':self.local_training_work,
            'diagnostic_input_binding':self.diagnostic_input_binding,
            'benign_update_exclusion_rate':sum(rejection_counts.values())/max(1,sum(submission_counts.values())) if c.attack=='none' else None,
            'source_fingerprints':source_map,
            'config_group_hash':hashlib.sha256(json.dumps(group_config,sort_keys=True).encode()).hexdigest(),
            **{k:c.to_dict()[k] for k in ('experiment_id','dataset','attack','defense','seed','num_clients',
            'participation_rate','malicious_fraction','dirichlet_alpha','identity_false_accept_rate',
            'identity_false_reject_rate','malicious_validator_fraction','duplicate_identity_fraction',
            'protocol_version','evidence_tier','model','detector','ablation','routing_budget')},
            'status':'completed','config_hash':hashlib.sha256(json.dumps(c.to_dict(),sort_keys=True).encode()).hexdigest(),
            'data_fingerprint':fingerprint(x,y),'test_fingerprint':fingerprint(tx,ty),
            'device':str(self.device),'task':'classification' if y.ndim == 1 else 'routing',
            'rounds_completed':c.rounds,'malicious_client_ids':sorted(malicious),
            'malicious_validator_ids':sorted(corrupt),'eligible_client_ids':enrolled,
            'malicious_updates_submitted':submitted,'malicious_updates_accepted':accepted_bad,
            'malicious_update_acceptance_rate':accepted_bad/submitted if submitted else 0.0,
            'final_accuracy':history[-1],'final_utility':history[-1],'accuracy_history':history,
            'utility_before_recovery_history':before_history,
            'targeted_attack_success_history':asrs,'targeted_asr_before_recovery_history':before_asrs,
            'mean_targeted_attack_success':float(np.mean(asrs)) if asrs else None,
            'clean_trigger_prediction_history':clean_trigger_history,
            'mean_clean_trigger_prediction_rate':float(np.mean(clean_trigger_history)) if clean_trigger_history else None,
            'recovery_utility_gain_sum':float(np.sum(np.array(history)-np.array(before_history))),
            'recovery_regret_sum':float(np.maximum(0,np.array(before_history)-np.array(history)).sum()),
            'transmitted_bytes':transmitted,'ledger_entries':len(ledger.entries),'ledger_bytes':ledger.serialized_size_bytes(),
            'recovery_count':recoveries,'wrongful_slashing_count':wrongful,
            'unsupported_rounds':unsupported,'trusted_server_samples':len(split.reference),
            'round_runtime_seconds':times,'runtime_seconds_total':sum(times),
            'worst_client_utility':min(utilities),'mean_client_utility':float(np.mean(utilities)),
            'test_clients_evaluated':len(utilities),
            'detection_true_positive_rate':tp/(tp+fn) if tp+fn else (0.0 if layered else None),
            'detection_false_positive_rate':fp/(fp+tn) if fp+tn else (0.0 if layered else None),
            'detection_false_negative_rate':fn/(tp+fn) if tp+fn else None,
            'detection_counts':{'tp':tp,'fp':fp,'fn':fn,'tn':tn},'detection_delay_rounds':delay,
            'enrollment_false_accepts':sum(d.error_kind=='false_accept' for d in enrollments),
            'enrollment_false_rejects':sum(d.error_kind=='false_reject' for d in enrollments),
            'duplicate_attempts':len(duplicates),'active_version':active_version,
            'final_stakes':{str(i):accountability.balance(i) for i in enrolled},
            'final_trust':{str(i):defense.trust_score(i) for i in enrolled} if layered else None,
            'validation_rule':c.vote_rule if layered else None,
            'client_test_utilities':utilities,
            'client_test_ids':[i for i,g in enumerate(test_partitions) if len(g)],
            'client_test_counts':[len(i) for i in test_partitions if len(i)],
            'lower_decile_client_utility':float(np.quantile(utilities,0.1)),
            'client_rejection_counts':rejection_counts,'client_submission_counts':submission_counts,
            'client_eligible_round_counts':eligibility_counts,
            'honest_validator_penalty_amount':sum(c.initial_stake-accountability.balance(i) for i in enrolled if i not in corrupt),
            'monitoring_quorum_failures':monitoring_quorum_failures,
            'recovery_policy':c.recovery_policy,'validator_behavior':c.validator_behavior}
        (self.output_dir/'chain-trace.json').write_text(json.dumps(self.trace,indent=2),encoding='utf-8')
        if c.archive_updates:
            torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},self.output_dir/'offchain-history/final.pt')
        (self.output_dir/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        self.logger.log('experiment_completed',c.rounds,summary)
        return summary
