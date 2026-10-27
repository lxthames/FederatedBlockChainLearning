import copy
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from threadpoolctl import threadpool_limits

from scripts.trust_alignment_v17 import study, analysis, behavior
from scripts.trust_alignment_v17.runner import BenchmarkRunner
from scripts.trust_repair_v16_fixed.integrity import IntegrityError
from scripts.trust_repair_v16_fixed.research.runner_v2 import BenchmarkRunner as ReferenceRunner
from scripts.trust_repair_v16_fixed.research.config import ExperimentConfig as ReferenceConfig


@pytest.fixture(autouse=True)
def threads():
    with threadpool_limits(limits=2):
        torch.set_num_threads(2)
        yield


def small(arm='current', attack='none'):
    return replace(study.config('fashion_mnist', arm, 11, attack), dataset='synthetic',
        num_classes=2, num_clients=20, participation_rate=.25, rounds=3, attack_start_round=2,
        natural_splits_path=None, device='cpu', server_samples_per_class=2,
        batch_size=16, dirichlet_alpha=100.)


def arrays():
    x = np.random.default_rng(71).random((1000, 1, 8, 8), dtype=np.float32)
    return x, (x[:, 0, 0, 0] > .5).astype('int64')


def factory():
    return nn.Sequential(nn.Flatten(), nn.Linear(64, 2))


def limits(value=10.):
    return {stage: {variant: {key: value for key in ('absolute', 'incremental', 'clean_excess')}
                    for variant in behavior.VARIANTS} for stage in behavior.STAGES}


def panel(value=0.):
    return dict(responses=5, complete=True,
        scores={variant: dict(absolute=value, incremental=value) for variant in behavior.VARIANTS},
        clean_excess=[0.]*5)


def clean_panels():
    return {(d, s): {(r, stage): panel() for r in range(1, 101) for stage in behavior.STAGES}
            for d in analysis.DATASETS for s in analysis.SEEDS}


def attack_panels():
    result = {}
    for d in analysis.DATASETS:
        for s in analysis.SEEDS:
            p = {(r, stage): panel() for r in range(1, 101) for stage in behavior.STAGES}
            for (r, stage), value in p.items():
                if r >= 11:
                    value['scores']['label_aligned']['absolute'] = 1.
            result[d, s] = p, [.1]*10+[.9]*90, 11
    return result


def run(c, folder, arm, threshold=None, evaluation=None):
    runner = BenchmarkRunner(c, folder)
    runner.challenge_arm, runner.challenge_thresholds = arm, threshold
    return runner.run_arrays(*arrays(), factory, evaluation)


def test_manifest_is_finite_clean_first_and_preserves_current_quorum():
    rows = study.manifest()
    assert len(rows) == 84
    assert len({r['config']['experiment_id'] for r in rows}) == 84
    assert all(r['role'] == 'passive' and r['attack'] == 'none' for r in rows[:6])
    assert all(r['role'] == 'passive' and r['attack'] == 'model_replacement' for r in rows[6:12])
    assert all(r['attack'] == 'none' for r in rows[12:36])
    assert all(r['config']['rounds'] == 100 for r in rows)
    for arm in ('current', 'original', 'candidate'):
        assert study.config('femnist', arm, 11).vote_rule == 'scaled_loss'
        assert study.config('femnist', arm, 11).aggregation_mass_policy == 'renormalized'
        assert study.config('femnist', arm, 11).trust_updates == 'fixed'


def test_probability_change_can_be_small_when_prediction_changes_completely():
    p = np.array([[[.51, .49], [.51, .49]], [[.49, .51], [.49, .51]]])
    assert behavior.contrast(p)[0, 1] == pytest.approx(.02)
    assert behavior.contrast(p, True)[0, 1] == 1.


def test_alignment_distinguishes_different_validator_coordinates():
    r = np.zeros((5, 5, 2))
    for v in range(5):
        r[v, v, 1] = 1.
    assert behavior.panel_score(r, False) == 1.
    assert behavior.panel_score(r, True) == 0.
    r[:2, 0, 1] = 1.
    assert behavior.panel_score(r, True) == 1.


def test_order_statistic_is_second_largest_not_fourth_largest():
    r = np.zeros((5, 1, 2)); r[0, 0, 1] = 1.
    assert behavior.panel_score(r, True) == 0.
    r[1, 0, 1] = .7
    assert behavior.panel_score(r, True) == .7


def test_inherited_behavior_remains_visible_in_absolute_score():
    before = {k: np.ones((5, 3, 2))*.8 for k in ('probability', 'label')}
    stats = behavior.scores(before, copy.deepcopy(before))
    assert stats['label_aligned'] == dict(absolute=.8, incremental=0.)
    p = panel(); p['scores'] = stats
    assert not behavior.decide(p, limits(.1)['screen'], 'label_aligned')


def test_insufficient_responses_fail_closed():
    p = panel(); p.update(responses=4, complete=False)
    assert not behavior.decide(p, limits()['screen'], 'label_aligned')
    with pytest.raises(ValueError, match='five'):
        behavior.panel_score(np.zeros((4, 3, 2)), True)


def test_four_of_five_clean_votes_required():
    p = panel(); p['clean_excess'] = [1., 0., 0., 0., 0.]
    assert behavior.decide(p, limits(.1)['screen'], 'label_aligned')
    p['clean_excess'][1] = 1.
    assert not behavior.decide(p, limits(.1)['screen'], 'label_aligned')


def test_nonfinite_scores_and_thresholds_are_rejected():
    with pytest.raises(ValueError):
        behavior.panel_score(np.full((5, 2, 2), np.nan), True)
    t = limits()['screen']; t['label_aligned']['absolute'] = float('nan')
    with pytest.raises(ValueError, match='threshold'):
        behavior.decide(panel(), t, 'label_aligned')


def test_leave_one_seed_out_calibration_is_reproducible_and_excludes_held_out_seed():
    clean = clean_panels()
    for d in analysis.DATASETS:
        for (r, stage), p in clean[d, 11].items():
            p['scores']['label_aligned']['absolute'] = 1e6
    a = analysis.calibrate(clean)
    assert a == analysis.calibrate(copy.deepcopy(clean))
    assert a['folds']['femnist']['11']['thresholds']['screen']['label_aligned']['absolute'] == 0.
    assert a['folds']['femnist']['17']['thresholds']['screen']['label_aligned']['absolute'] == 1e6
    assert a['folds']['femnist']['11']['training_seeds'] == [17, 23]
    clean.pop(('femnist', 23))
    with pytest.raises(ValueError, match='six'):
        analysis.calibrate(clean)


def test_fixed_diagnostic_gate_and_all_seed_requirement():
    clean, attacked = clean_panels(), attack_panels()
    calibration = analysis.calibrate(clean)
    effective = {key: True for key in attacked}
    report = analysis.diagnostic_report(clean, attacked, calibration, effective)
    assert report['passed']
    assert all(r['recall_improvement'] == 1. for r in report['primary_checks'])
    effective['femnist', 23] = False
    assert not analysis.diagnostic_report(clean, attacked, calibration, effective)['passed']


def test_clean_veto_limit_applies_to_both_datasets_and_every_seed():
    clean, attacked = clean_panels(), attack_panels()
    calibration = analysis.calibrate(clean)
    for r in range(1, 5):
        clean['fashion_mnist', 23][r, 'screen']['scores']['label_aligned']['absolute'] = 1.
    assert not analysis.diagnostic_report(clean, attacked, calibration, {k: True for k in attacked})['passed']


def test_empty_high_asr_set_is_not_a_defense_success():
    clean, attacked = clean_panels(), attack_panels()
    panels, _, onset = attacked['femnist', 11]
    attacked['femnist', 11] = panels, [.1]*100, onset
    report = analysis.diagnostic_report(clean, attacked, analysis.calibrate(clean), {k: True for k in attacked})
    assert not report['passed']
    assert report['primary_checks'][0]['recall_improvement'] is None


@pytest.mark.parametrize('attack', ['none', 'model_replacement'])
def test_passive_telemetry_preserves_every_small_reference_model_hash(tmp_path, attack):
    c = small(attack=attack)
    values = c.to_dict(); values.update(protocol_version='trust-repair-v16-fixed', experiment_id='reference')
    reference = ReferenceRunner(ReferenceConfig(**values), tmp_path/'reference')
    expected = reference.run_arrays(*arrays(), factory)
    actual = run(c, tmp_path/'passive', 'passive')
    left = study.read(tmp_path/'reference/chain-trace.json')
    right = study.read(tmp_path/'passive/chain-trace.json')
    assert [r['model_hash'] for r in left] == [r['model_hash'] for r in right]
    assert actual['final_utility'] == expected['final_utility']
    assert actual['local_training_work'] == expected['local_training_work']
    events = [json.loads(l) for l in (tmp_path/'passive/events.jsonl').read_text().splitlines()]
    assert len([e for e in events if e['event_type'] == 'alignment_panel']) == 6
    assert len(list((tmp_path/'passive/behavior').glob('*.npz'))) == 30


def test_candidate_validated_commit_matches_exact_scored_model(tmp_path):
    c = small('candidate')
    result = run(c, tmp_path/'candidate', 'candidate', limits())
    reference = run(c, tmp_path/'current', 'current')
    assert result['accuracy_history'] == reference['accuracy_history']
    events = [json.loads(l) for l in (tmp_path/'candidate/events.jsonl').read_text().splitlines()]
    checks = [e['payload'] for e in events if e['event_type'] == 'model_integrity']
    assert all(e['validated_hash'] == e['committed_hash'] == e['expected_hash'] for e in checks)


def test_evaluation_labels_cannot_change_gate_decisions_or_models(tmp_path):
    c = small('candidate'); x, y = arrays()
    run(c, tmp_path/'left', 'candidate', limits(), (x[:30], y[:30]))
    run(c, tmp_path/'right', 'candidate', limits(), (x[:30], 1-y[:30]))
    hashes = lambda folder: [r['model_hash'] for r in study.read(folder/'chain-trace.json')]
    assert hashes(tmp_path/'left') == hashes(tmp_path/'right')
    decisions = lambda folder: [json.loads(l)['payload'] for l in (folder/'events.jsonl').read_text().splitlines()
                               if json.loads(l)['event_type'] == 'alignment_decision']
    assert decisions(tmp_path/'left') == decisions(tmp_path/'right')


def test_inference_mutation_is_detected(tmp_path):
    class Mutable(nn.Module):
        def __init__(self):
            super().__init__(); self.weight = nn.Parameter(torch.zeros(1))
        def forward(self, x):
            self.weight.add_(1.)
            return torch.zeros((len(x), 2))
    runner = SimpleNamespace(config=SimpleNamespace(seed=11, evaluation_batch_size=32, num_classes=2),
        challenge_arm='current', challenge_thresholds=None, validation_work={}, device='cpu')
    obj = behavior.AlignmentRound(runner, factory(), None, None, None, 1, [], [])
    with pytest.raises(IntegrityError, match='alignment inference'):
        obj.probabilities(Mutable(), np.zeros((4, 1, 8, 8), dtype='float32'), 2)


def test_behavior_engine_does_not_reference_attacker_or_evaluation_fields():
    from pathlib import Path
    source = Path(behavior.__file__).read_text()
    for forbidden in ('backdoor_target_label', 'is_malicious', 'malicious_client_ids', 'targeted_attack_success',
                      'targeted_asr', '.test', 'config.attack'):
        assert forbidden not in source


def test_candidate_cannot_start_without_frozen_calibration(tmp_path):
    with pytest.raises(ValueError, match='calibration'):
        run(small('candidate'), tmp_path, 'candidate')


def test_completion_seal_detects_npz_tampering(tmp_path):
    folder = tmp_path/'cell'; folder.mkdir()
    np.savez_compressed(folder/'scores.npz', counts=np.array([1, 2]))
    seal = study.sealed_files(folder)
    study.dump(folder/'completion.json', seal)
    assert study.sealed_files(folder) == seal
    np.savez_compressed(folder/'scores.npz', counts=np.array([2, 1]))
    assert study.sealed_files(folder) != seal


def test_blocked_qualification_cannot_start_development(tmp_path, monkeypatch):
    monkeypatch.setattr(study, 'JOB', tmp_path)
    study.dump(tmp_path/'qualification.json', dict(passed=False))
    with pytest.raises(ValueError, match='gates'):
        study.assert_qualification()


def test_corruption_at_global_commit_stops_the_new_runner(tmp_path, monkeypatch):
    from scripts.trust_alignment_v17 import runner as module
    c = small('candidate'); runner = BenchmarkRunner(c, tmp_path)
    runner.challenge_arm, runner.challenge_thresholds = 'candidate', limits()
    original = module._apply_delta
    def corrupted(model, delta):
        original(model, delta)
        if hasattr(runner, 'challenge') and model is runner.challenge.parent:
            with torch.no_grad():
                next(model.parameters()).view(-1)[0] += 100.
    monkeypatch.setattr(module, '_apply_delta', corrupted)
    with pytest.raises(IntegrityError, match='global commit'):
        runner.run_arrays(*arrays(), factory)


def test_candidate_hold_has_no_double_unsupported_count_or_inclusion(tmp_path):
    c = small('candidate')
    result = run(c, tmp_path, 'candidate', limits(0.))
    trace = study.read(tmp_path/'chain-trace.json')
    decisions = [json.loads(l) for l in (tmp_path/'events.jsonl').read_text().splitlines()
                 if json.loads(l)['event_type'] == 'alignment_decision']
    held = [e for e in decisions if not e['payload']['passed']]
    assert held
    assert result['unsupported_rounds'] <= c.rounds
    for event in held:
        assert trace[event['round_index']]['accepted_client_ids'] == []
        assert trace[event['round_index']]['model_hash'] == trace[event['round_index']-1]['model_hash']


def test_end_to_end_targets_reject_collapse_and_ineffective_controls():
    records = []
    for d in analysis.DATASETS:
        for s in analysis.SEEDS:
            for arm in study.ARMS:
                for attack in study.ATTACKS:
                    targeted = attack == 'model_replacement'
                    asr = .1 if arm == 'candidate' else .9
                    records.append(dict(dataset=d, seed=s, arm=arm, attack=attack,
                        experiment_id=f'{d}-{s}-{arm}-{attack}', final_utility=.8,
                        post_onset_mean_asr=asr if targeted else None,
                        endpoint_asr=asr if targeted else None,
                        post_onset_mean_clean_trigger_rate=.01 if attack == 'none' else None,
                        lower_decile_client_utility=.5, honest_update_exclusion_rate=.1,
                        mean_post_onset_malicious_weight=.01, malicious_update_acceptance_rate=.2,
                        unsupported_rounds=0, honest_validator_penalty_amount=0,
                        validation_work={}, runtime_seconds_total=1., transmitted_bytes=100,
                        artifact_bytes=100))
    pair = lambda a,b: dict(matched=bool(a and b))
    assert analysis.end_to_end_report(records, pair, pair)['passed']
    changed = copy.deepcopy(records)
    next(r for r in changed if r['dataset']=='femnist' and r['arm']=='candidate'
         and r['attack']=='model_replacement')['final_utility'] = .05
    assert not analysis.end_to_end_report(changed, pair, pair)['passed']
    changed = copy.deepcopy(records)
    next(r for r in changed if r['arm']=='clip_only' and r['attack']=='model_replacement')['post_onset_mean_asr'] = .01
    assert not analysis.end_to_end_report(changed, pair, pair)['passed']


def test_process_liveness_checks_current_and_absent_process_without_signalling():
    import os
    assert study.process_alive(os.getpid())
    assert not study.process_alive(2**30)
