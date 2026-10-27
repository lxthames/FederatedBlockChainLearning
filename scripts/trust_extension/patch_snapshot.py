"""One-time, checked transformations of the isolated v2 snapshot."""
from pathlib import Path

BASE = Path(__file__).parent/'research'


def edit(name, old, new):
    path = BASE/name
    text = path.read_text(encoding='utf-8')
    if text.count(old) != 1:
        raise RuntimeError(f'{name}: expected one anchor, found {text.count(old)}: {old[:70]}')
    path.write_text(text.replace(old, new), encoding='utf-8')


def main():
    edit('config.py', '    def __post_init__(self) -> None:', '''    natural_splits_path: str | None = None
    recovery_policy: str = "protocol"
    intervention_round: int = 20
    intervention_checkpoint: int = 15
    validator_behavior: str = "approve_bad_reject_good"
    adaptive_clean_tolerance: float = 0.02
    adaptive_norm_ratio: float = 1.5

    def __post_init__(self) -> None:
        if self.recovery_policy not in {'protocol','continue','scheduled_plain','scheduled_accountable'}:
            raise ValueError('invalid recovery policy')
        if self.validator_behavior not in {'approve_bad_reject_good','false_recovery','withhold','honest'}:
            raise ValueError('invalid validator behavior')
        if not 0 <= self.intervention_checkpoint < self.intervention_round:
            raise ValueError('invalid recovery intervention interval')
        if self.adaptive_clean_tolerance < 0 or self.adaptive_norm_ratio <= 0:
            raise ValueError('invalid adaptive attack limits')''')
    edit('config.py', 'protocol_version: str = "trust-v2"', 'protocol_version: str = "trust-extension-v1"')
    edit('config.py', 'self.protocol_version != "trust-v2"', 'self.protocol_version != "trust-extension-v1"')
    edit('models.py', 'if dataset == "fashion_mnist":', 'if dataset in {"fashion_mnist", "femnist"}:')
    edit('evm.py', 'Path(__file__).resolve().parents[2]', 'Path(__file__).resolve().parents[3]')
    edit('splits.py', 'def make_splits(labels, config, external_test=False):', 'def make_standard_splits(labels, config, external_test=False):')
    with (BASE/'splits.py').open('a', encoding='utf-8') as out:
        out.write('''

def make_splits(labels, config, external_test=False):
    if not config.natural_splits_path:
        return make_standard_splits(labels, config, external_test)
    import json
    from pathlib import Path
    if not external_test:
        raise ValueError('natural splits require official external test arrays')
    data = json.loads(Path(config.natural_splits_path).read_text())
    if data['seed'] != config.seed or len(data['train']) != config.num_clients:
        raise ValueError('natural split seed/client mismatch')
    groups = [np.asarray(g, dtype=np.int64) for g in data['train']+data['validation']+[data['reference']]]
    all_indices = np.concatenate(groups)
    if any(not len(g) for g in groups) or len(all_indices) != len(labels) or not np.array_equal(np.sort(all_indices),np.arange(len(labels))):
        raise ValueError('natural splits overlap, omit data, or contain invalid indices')
    if set(data['writers']) & set(data['reference_writers']):
        raise ValueError('reference writer overlaps enrolled writers')
    split = DataSplit(groups[:config.num_clients],groups[config.num_clients:2*config.num_clients],groups[-1],np.array([],dtype=np.int64))
    split.natural_test_clients = [np.asarray(g,dtype=np.int64) for g in data['test_clients']]
    return split
''')
    edit('splits.py', '    train_classes=train_labels if train_labels.ndim==1 else train_labels.argmax(1)', '''    if hasattr(split, 'natural_test_clients'):
        groups = split.natural_test_clients
        if not np.array_equal(np.sort(np.concatenate(groups)), np.arange(len(test_labels))):
            raise ValueError('natural test shards overlap or omit examples')
        return groups
    train_classes=train_labels if train_labels.ndim==1 else train_labels.argmax(1)''')
    edit('runner_v2.py', 'from .detector import TrendDetector, AdjacentDetector', 'from .detector import TrendDetector, AdjacentDetector, DetectionResult\nfrom ..fedval import FedVal\nfrom ..adaptive import constrained_replacement')
    edit('runner_v2.py', 'from .contracts import ClientUpdate, AggregationResult', 'from .contracts import ClientUpdate, AggregationResult, VoteDecision')
    edit('runner_v2.py', "        detector = detector_cls(c.detector_window, c.relative_drop_threshold, c.slope_threshold)", '''        detector = detector_cls(c.detector_window, c.relative_drop_threshold, c.slope_threshold)
        fedval = FedVal(self, x[split.reference], y[split.reference], int(y.max())+1) if c.defense == 'fedval' else None
        validator_monitors = {}
        intervention_signers = set()
        monitoring_quorum_failures = 0
        rejection_counts = {i: 0 for i in enrolled}
        submission_counts = {i: 0 for i in enrolled}
        eligibility_counts = {i: 0 for i in enrolled}''')
    edit('runner_v2.py', '            updates = []', '''            for i in eligible:
                eligibility_counts[i] += 1
            updates = []''')
    edit('runner_v2.py', "poison=attack_active and client in malicious and c.attack == 'model_replacement')\n                updates.append(ClientUpdate(client, _state_delta(local, model), len(indices), client in malicious))", '''poison=attack_active and client in malicious and c.attack in {'model_replacement','adaptive_model_replacement'})
                delta = _state_delta(local, model)
                metadata = {}
                if attack_active and client in malicious and c.attack == 'adaptive_model_replacement':
                    honest_local = copy.deepcopy(model)
                    self._train(honest_local, x[indices], y[indices], rnd, client)
                    delta, metadata = constrained_replacement(self, model, _state_delta(honest_local, model), delta, x[indices], y[indices])
                updates.append(ClientUpdate(client, delta, len(indices), client in malicious, metadata))
                submission_counts[client] += 1''')
    edit('runner_v2.py', "else _attack(c.attack)", "else _attack('none' if c.attack == 'adaptive_model_replacement' else c.attack)")
    edit('runner_v2.py', "                        if validator in corrupt:\n                            score = baseline if u.client_id in malicious else baseline-1.0", '''                        if validator in corrupt and c.validator_behavior == 'withhold':
                            continue
                        if validator in corrupt and c.validator_behavior == 'approve_bad_reject_good':
                            score = baseline if u.client_id in malicious else baseline-1.0''')
    edit('runner_v2.py', '                    decisions[u.client_id] = VoteAggregator().decide(votes, policy.quorum)', '''                    decisions[u.client_id] = VoteAggregator().decide(votes, policy.quorum) if votes else VoteDecision(False,policy.quorum,(),0.0)''')
    edit('runner_v2.py', "                baseline = _defense(c.defense,byzantine_bound,trusted,server_vectors)\n                if c.defense == 'multi_krum':\n                    baseline = MultiKrumDefense(byzantine_bound,max(1,len(attacked)-2*byzantine_bound-2))\n                result = baseline.aggregate(model.state_dict(),attacked,None)", '''                if fedval:
                    result = fedval.aggregate(model, attacked)
                    self.logger.log('fedval_weights', rnd, dict(result.diagnostics))
                else:
                    baseline = _defense(c.defense,byzantine_bound,trusted,server_vectors)
                    if c.defense == 'multi_krum':
                        baseline = MultiKrumDefense(byzantine_bound,max(1,len(attacked)-2*byzantine_bound-2))
                    result = baseline.aggregate(model.state_dict(),attacked,None)''')
    edit('runner_v2.py', "targeted = c.attack == 'model_replacement' and y.ndim == 1", "targeted = c.attack in {'model_replacement','adaptive_model_replacement'} and y.ndim == 1")
    edit('runner_v2.py', '            parent = active_version', '''            intervention_signers.update(signers if rnd > c.intervention_checkpoint else [])
            for u in attacked:
                if u.client_id not in result.accepted_client_ids:
                    rejection_counts[u.client_id] += 1
            parent = active_version''')
    edit('runner_v2.py', "                        initial_score = np.median([1.0 if i in corrupt else self._score(previous,x[split.validation[i]],y[split.validation[i]]) for i in panel])\n                        detector.observe(float(initial_score))", '''                        validator_monitors = {i: detector_cls(c.detector_window,c.relative_drop_threshold,c.slope_threshold) for i in panel}
                        initial = [self._score(previous,x[split.validation[i]],y[split.validation[i]]) for i in panel]
                        for i, score in zip(panel, initial):
                            validator_monitors[i].observe(score)
                        detector.observe(float(np.median(initial)))''')
    edit('runner_v2.py', "                    monitoring = float(np.median([1.0 if i in corrupt else self._score(model,x[split.validation[i]],y[split.validation[i]]) for i in panel]))", '''                    reported = {}
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
                    monitoring = float(np.median(list(reported.values())))''')
    edit('runner_v2.py', "                    self.logger.log('monitor',rnd,{'panel':panel,'score':monitoring,'window_start_version':panel_start})\n                    if detection.detected:\n                        penalized = sorted(window_signers) if c.ablation != 'no_penalties' else []", '''                    quorum = math.ceil(2*len(panel)/3)
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
                    self.logger.log('monitor',rnd,{'panel':panel,'score':monitoring,'window_start_version':panel_start,
                        'reports':reported,'affirmative_recovery_attesters':attestations,'quorum':quorum})
                    if detection.detected:
                        penalized = sorted(window_signers) if c.ablation != 'no_penalties' and c.recovery_policy != 'scheduled_plain' else []''')
    edit('runner_v2.py', "'panel':panel,\n                            'penalties_enabled':c.ablation != 'no_penalties'", "'panel':attestations,\n                            'penalties_enabled':c.ablation != 'no_penalties' and c.recovery_policy != 'scheduled_plain'")
    edit('runner_v2.py', "            'final_stakes':{str(i):accountability.balance(i) for i in enrolled}}", '''            'final_stakes':{str(i):accountability.balance(i) for i in enrolled},
            'client_test_utilities':utilities,
            'client_test_counts':[len(i) for i in test_partitions if len(i)],
            'lower_decile_client_utility':float(np.quantile(utilities,0.1)),
            'client_rejection_counts':rejection_counts,'client_submission_counts':submission_counts,
            'client_eligible_round_counts':eligibility_counts,
            'honest_validator_penalty_amount':sum(c.initial_stake-accountability.balance(i) for i in enrolled if i not in corrupt),
            'monitoring_quorum_failures':monitoring_quorum_failures,
            'recovery_policy':c.recovery_policy,'validator_behavior':c.validator_behavior}''')


if __name__ == '__main__':
    main()
