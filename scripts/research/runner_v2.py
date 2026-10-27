"""Disjoint evaluation, independent randomness and auditable recovery protocol."""
import copy
import hashlib
import json
import math
import platform
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .runner_helpers import _attack, _defense, _apply_delta, _state_delta, _update_bytes
from .attacks import AttackContext, LowAndSlowAttack
from .contracts import ClientUpdate, AggregationResult
from .defenses import TwoLayerDefense, AggregationContext, flatten_update, MultiKrumDefense
from .committee import CommitteePolicy
from .validation import LocalValidator, VoteAggregator
from .identity import IdentityOracle
from .ledger import AppendOnlyRecoveryLedger
from .accountability import AccountabilityManager
from .detector import TrendDetector, AdjacentDetector
from .events import EventLogger
from .splits import stream, make_splits, fingerprint, matched_test_shards


def model_hash(model):
    return fingerprint(*[x.detach().cpu().numpy() for x in model.state_dict().values()])


class BenchmarkRunner:
    def __init__(self, config, output_dir):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if (self.output_dir / 'events.jsonl').exists():
            raise FileExistsError('refusing to overwrite an existing run')
        self.logger = EventLogger(self.output_dir / 'events.jsonl', config.experiment_id, config.seed)
        self.device = torch.device('cuda' if config.device == 'auto' and torch.cuda.is_available()
                                   else ('cpu' if config.device == 'auto' else config.device))
        self.trace = []

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

    @staticmethod
    def _apply_trigger(features):
        triggered = features.copy()
        if triggered.ndim == 4:
            triggered[:, :, -2:, -2:] = 1.0
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
        optimizer = torch.optim.SGD(model.parameters(), lr=c.learning_rate)
        model.train()
        for _ in range(c.local_epochs):
            for bx, by in loader:
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

    def run_arrays(self, features, labels, model_factory, test_arrays=None):
        c = self.config
        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(labels, dtype=np.int64 if labels.ndim == 1 else np.float32)
        if len(x) != len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('incompatible or non-finite input arrays')
        if c.attack == 'model_replacement' and y.ndim != 1:
            raise ValueError('targeted model replacement requires classification labels')
        if y.ndim == 1 and not 0 <= c.backdoor_target_label < int(y.max())+1:
            raise ValueError('target label outside task')
        split = make_splits(y, c, external_test=test_arrays is not None)
        tx, ty = (x[split.test], y[split.test]) if test_arrays is None else test_arrays
        tx = np.asarray(tx, dtype=np.float32)
        ty = np.asarray(ty, dtype=y.dtype)
        classes = y if y.ndim == 1 else y.argmax(1)
        torch.manual_seed(c.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
        model = model_factory().to(self.device)
        self.logger.log('environment', 0, {'python_version': platform.python_version(),
            'torch_version': torch.__version__, 'numpy_version': np.__version__,
            'device': str(self.device), 'gpu': torch.cuda.get_device_name(0) if self.device.type == 'cuda' else None,
            'config': c.to_dict(), 'data_fingerprint': fingerprint(x, y),
            'test_fingerprint': fingerprint(tx, ty)})
        source_hash=hashlib.sha256(b''.join(p.read_bytes() for p in sorted(Path(__file__).parent.glob('*.py')))).hexdigest()
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
        defense = TwoLayerDefense(c.initial_trust, c.trust_reward, c.trust_penalty) if layered else None
        detector_cls = TrendDetector if c.detector == 'trend' else AdjacentDetector
        detector = detector_cls(c.detector_window, c.relative_drop_threshold, c.slope_threshold)
        history, before_history, asrs, before_asrs, times = [], [], [], [], []
        clean_trigger_history=[]
        submitted = accepted_bad = recoveries = wrongful = transmitted = unsupported = 0
        alie_reference_shortfalls = 0
        tp = fp = fn = tn = 0
        panel, window_rounds, window_signers = [], [], set()
        panel_start = 0
        active_version = 0
        commits = []
        monitor_values = []
        first_attack = delay = None
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
            transmitted += sum(v.numel()*v.element_size() for v in model.state_dict().values())*len(participants)
            eligible = [i for i in enrolled if (not layered or
                (accountability.balance(i) >= c.minimum_stake and defense.trust_score(i) >= c.minimum_trust))]
            updates = []
            attack_active = rnd >= c.attack_start_round and c.attack != 'none'
            for client in participants:
                if layered and defense.trust_score(client) < c.trust_floor:
                    continue
                indices = split.train[client]
                local = copy.deepcopy(model)
                self._train(local, x[indices], y[indices], rnd, client,
                            poison=attack_active and client in malicious and c.attack == 'model_replacement')
                updates.append(ClientUpdate(client, _state_delta(local, model), len(indices), client in malicious))
            bad_count = sum(u.is_malicious for u in updates) if attack_active else 0
            active = bad_count > 0
            if active and first_attack is None:
                first_attack = rnd
            attack = LowAndSlowAttack(c.gradual_max_norm_ratio,c.gradual_ramp_rounds) if c.attack == 'low_and_slow' else _attack(c.attack)
            honest = tuple(u for u in updates if not u.is_malicious)
            attacked = []
            for u in updates:
                if u.is_malicious and attack_active:
                    attack_round = rnd-c.attack_start_round+1
                    u = attack.apply(u, AttackContext(attack_round,c.seed,len(updates),max(1,bad_count),honest))
                    if u.metadata.get('attack_reason') == 'insufficient_reference_peers':
                        alie_reference_shortfalls += 1
                attacked.append(u)
            submitted += bad_count
            transmitted += sum(_update_bytes(u) for u in attacked)
            decisions, candidate_trace = {}, []
            if layered:
                for u in attacked:
                    pool = [i for i in eligible if i != u.client_id]
                    policy = CommitteePolicy().for_population(len(pool))
                    if not policy.consensus_supported:
                        continue
                    members = sorted(stream(c.seed,'committee',rnd,u.client_id).choice(pool,policy.size,replace=False).tolist())
                    votes = []
                    candidate = copy.deepcopy(model)
                    _apply_delta(candidate,dict(u.state))
                    for validator in members:
                        vi = split.validation[validator]
                        baseline = self._score(model,x[vi],y[vi])
                        score = self._score(candidate,x[vi],y[vi])
                        if validator in corrupt:
                            score = baseline if u.client_id in malicious else baseline-1.0
                        vote = LocalValidator(validator,c.minimum_validation_delta).vote(u.client_id,baseline,score,validator in corrupt)
                        votes.append(vote)
                    decisions[u.client_id] = VoteAggregator().decide(votes, policy.quorum)
                    delta_hash = fingerprint(*[v.numpy() for v in u.state.values()])
                    candidate_trace.append({'client_id':u.client_id,'candidate_hash':delta_hash,
                        'committee':members, 'quorum':policy.quorum,
                        'votes':[{'validator_id':v.validator_id,'approve':v.approve} for v in votes],
                        'accepted':decisions[u.client_id].accepted})
                    transmitted += _update_bytes(u)*len(members) + len(members)*96
                supported = [u for u in attacked if u.client_id in decisions]
                if supported and any(d.accepted for d in decisions.values()):
                    result = defense.aggregate(model.state_dict(),supported,AggregationContext(rnd,decisions))
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
                baseline = _defense(c.defense,byzantine_bound,trusted,server_vectors)
                if c.defense == 'multi_krum':
                    baseline = MultiKrumDefense(byzantine_bound,max(1,len(attacked)-2*byzantine_bound-2))
                result = baseline.aggregate(model.state_dict(),attacked,None)
            _apply_delta(model,dict(result.state))
            before = self._score(model,tx,ty)
            before_history.append(before)
            targeted = c.attack == 'model_replacement' and y.ndim == 1
            trigger_control = c.attack == 'none' and y.ndim == 1
            target_mask = ty != c.backdoor_target_label if targeted or trigger_control else None
            def asr():
                if not (targeted or trigger_control) or not np.any(target_mask):
                    return None
                return self._score(model,self._apply_trigger(tx[target_mask]),np.full(int(target_mask.sum()),c.backdoor_target_label,dtype=np.int64))
            pre_asr = asr()
            before_asrs.append(pre_asr)
            signers = sorted({s for i in result.accepted_client_ids for s in decisions[i].affirmative_signers}) if layered else []
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
                        initial_score = np.median([1.0 if i in corrupt else self._score(previous,x[split.validation[i]],y[split.validation[i]]) for i in panel])
                        detector.observe(float(initial_score))
                if panel:
                    window_rounds.append(rnd)
                    window_signers.update(signers)
                    monitoring = float(np.median([1.0 if i in corrupt else self._score(model,x[split.validation[i]],y[split.validation[i]]) for i in panel]))
                    transmitted += len(panel)*96
                    monitor_values.append(monitoring)
                    detection = detector.observe(monitoring)
                    self.logger.log('monitor',rnd,{'panel':panel,'score':monitoring,'window_start_version':panel_start})
                    if detection.detected:
                        penalized = sorted(window_signers) if c.ablation != 'no_penalties' else []
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
                            'penalties_enabled':c.ablation != 'no_penalties','trigger':detection.reason}
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
                'attack_execution':[{ 'client_id':u.client_id, **dict(u.metadata)}
                    for u in attacked if u.is_malicious and u.metadata],
                'accepted_client_ids':list(result.accepted_client_ids),
                'accuracy':history[-1],'utility_before_recovery':before,
                'targeted_attack_success':post_asr,'targeted_asr_before_recovery':pre_asr,
                'committee_decisions':{str(i):asdict(d) for i,d in decisions.items()},
                'detection':asdict(detection) if detection else None,
                'runtime_seconds':times[-1], 'active_version':active_version})
        # Independent held-out test shards for fairness; never private validation scores.
        test_partitions = matched_test_shards(y,ty,split,c.seed)
        utilities = [self._score(model,tx[i],ty[i]) for i in test_partitions if len(i)]
        group_config = {k:v for k,v in c.to_dict().items() if k not in {'experiment_id','seed','device','rpc_url','deployment_path'}}
        summary = {'configuration':c.to_dict(),'source_hash':source_hash,
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
            'alie_reference_shortfalls':alie_reference_shortfalls,
            'alie_shortfall_policy':'submit_unmodified_local_update_when_fewer_than_two_reference_peers',
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
            'final_stakes':{str(i):accountability.balance(i) for i in enrolled}}
        (self.output_dir/'chain-trace.json').write_text(json.dumps(self.trace,indent=2),encoding='utf-8')
        (self.output_dir/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        self.logger.log('experiment_completed',c.rounds,summary)
        return summary
