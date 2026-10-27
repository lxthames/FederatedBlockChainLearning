"""Clean-only calibration and fixed diagnostic/development decisions."""
import json
from pathlib import Path

import numpy as np

from .behavior import VARIANTS, METRICS, STAGES, QUANTILE, decide

SEEDS = (11, 17, 23)
DATASETS = ('fashion_mnist', 'femnist')


def read_panels(folder, rounds=100):
    values = {}
    for line in (Path(folder)/'events.jsonl').open(encoding='utf-8'):
        event = json.loads(line)
        if event['event_type'] != 'alignment_panel':
            continue
        key = (event['round_index'], event['payload']['stage'])
        if key in values:
            raise ValueError('duplicate behavioral panel')
        values[key] = event['payload']
    if set(values) != {(r, stage) for r in range(1, rounds+1) for stage in STAGES}:
        raise ValueError('missing behavioral panels')
    if any(not p['complete'] or p['responses'] != 5 for p in values.values()):
        raise ValueError('insufficient behavioral responses')
    return values


def calibrate(clean):
    """Only three clean trajectories per dataset are accepted; no attack argument."""
    expected = {(d, s) for d in DATASETS for s in SEEDS}
    if set(clean) != expected:
        raise ValueError('six complete clean calibration trajectories required')
    folds = {}
    for dataset in DATASETS:
        folds[dataset] = {}
        for held_out in SEEDS:
            stages = {}
            for stage in STAGES:
                panels = [p for seed in SEEDS if seed != held_out
                          for (rnd, st), p in clean[dataset, seed].items() if st == stage]
                if len(panels) != 200 or not all(p['complete'] for p in panels):
                    raise ValueError('incomplete clean calibration fold')
                thresholds = {}
                for variant in VARIANTS:
                    metrics = {key: [p['scores'][variant][key] for p in panels] for key in METRICS}
                    metrics['clean_excess'] = [v for p in panels for v in p['clean_excess']]
                    if any(not np.isfinite(vals).all() or min(vals) < 0 for vals in metrics.values()):
                        raise ValueError('invalid clean calibration values')
                    thresholds[variant] = {key: float(np.quantile(vals, QUANTILE, method='higher'))
                                           for key, vals in metrics.items()}
                stages[stage] = thresholds
            folds[dataset][str(held_out)] = dict(training_seeds=[s for s in SEEDS if s != held_out],
                                               thresholds=stages)
    return dict(quantile=QUANTILE, method='higher', folds=folds,
                source='other-two-seed clean current-quorum trajectories only',
                independent_fpr_guarantee=False)


def veto_rounds(panels, thresholds, variant):
    rounds = sorted({r for r, stage in panels})
    return {r: not all(decide(panels[r, stage], thresholds[stage], variant)
                       for stage in STAGES) for r in rounds}


def diagnostic_report(clean, attacked, calibration, effectiveness):
    """ASR labels are read only here, after clean thresholds and panels are fixed."""
    clean_checks, attack_checks, primary = [], [], []
    for dataset in DATASETS:
        for seed in SEEDS:
            limits = calibration['folds'][dataset][str(seed)]['thresholds']
            for variant in VARIANTS:
                veto = veto_rounds(clean[dataset, seed], limits, variant)
                clean_checks.append(dict(dataset=dataset, seed=seed, variant=variant,
                                         vetoes=sum(veto.values()), passed=sum(veto.values()) <= 3))
            panels, asr, onset = attacked[dataset, seed]
            high = [r for r in range(onset, 101) if asr[r-1] > .5]
            recalls = {}
            for variant in VARIANTS:
                veto = veto_rounds(panels, limits, variant)
                recall = float(np.mean([veto[r] for r in high])) if high else None
                recalls[variant] = recall
                alarm = next((r for r in range(onset, 101) if veto[r]), None)
                attack_checks.append(dict(dataset=dataset, seed=seed, variant=variant,
                    high_asr_rounds=len(high), high_asr_recall=recall, first_post_onset_alarm=alarm,
                    post_onset_vetoes=sum(veto[r] for r in range(onset, 101))))
            if dataset == 'femnist':
                effective = effectiveness.get((dataset, seed)) is True
                base, candidate = recalls['probability_independent'], recalls['label_aligned']
                improvement = candidate-base if base is not None and candidate is not None else None
                primary.append(dict(dataset=dataset, seed=seed, effective_control_attack=effective,
                    high_asr_rounds=len(high), recall_improvement=improvement,
                    passed=bool(effective and improvement is not None and improvement >= .20-1e-12)))
    clean_passed = all(row['passed'] for row in clean_checks if row['variant'] == 'label_aligned')
    passed = clean_passed and len(primary) == 3 and all(row['passed'] for row in primary)
    return dict(passed=passed, candidate='label_aligned', reference='probability_independent',
                clean_checks=clean_checks, attack_checks=attack_checks, primary_checks=primary,
                clean_passed=clean_passed, publication_ready=False, retrained_defense_evidence=False,
                rule='all six clean seeds <=3 vetoes; each effective FEMNIST seed >=20pp recall gain on ASR>50% rounds; missing/ineffective/empty positives fail closed')


def end_to_end_report(records, method_pair, condition_pair):
    index = {(r['dataset'], r['arm'], r['seed'], r['attack']): r for r in records}
    rows, groups, effective = [], [], []
    for d in DATASETS:
        for s in SEEDS:
            for attack in ('none', 'low_and_slow', 'model_replacement'):
                proposed = index.get((d, 'candidate', s, attack))
                base = index.get((d, 'current', s, attack))
                matched = bool(proposed and base and method_pair(proposed, base)['matched'])
                if matched and attack != 'none':
                    matched = all(condition_pair(index.get((d, arm, s, 'none')), row)['matched']
                                  for arm, row in (('candidate', proposed), ('current', base)))
                utility_delta = proposed['final_utility']-base['final_utility'] if matched else None
                passed = bool(matched and proposed['honest_validator_penalty_amount'] == 0)
                passed = passed and utility_delta >= -.02-1e-12
                if attack == 'none':
                    passed = passed and proposed['unsupported_rounds'] <= 3
                if attack == 'model_replacement' and d == 'femnist':
                    passed = passed and all(proposed[k] <= .2+1e-12 and base[k]-proposed[k] >= .2-1e-12
                                            for k in ('post_onset_mean_asr', 'endpoint_asr'))
                rows.append(dict(dataset=d, seed=s, attack=attack, matched=matched,
                                 utility_difference=utility_delta, passed=bool(passed)))
    for d in DATASETS:
        for arm in ('clip_only', 'current', 'original', 'candidate'):
            for attack in ('none', 'low_and_slow', 'model_replacement'):
                selected = [index.get((d, arm, s, attack)) for s in SEEDS]
                complete = all(selected)
                if complete and attack != 'none':
                    complete = all(condition_pair(index.get((d, arm, r['seed'], 'none')), r)['matched'] for r in selected)
                mean = lambda k: float(np.mean([r[k] for r in selected])) if complete else None
                groups.append(dict(dataset=d, arm=arm, attack=attack, complete=bool(complete),
                    accuracy=mean('final_utility'), unsupported=mean('unsupported_rounds'),
                    lower_decile_utility=mean('lower_decile_client_utility'),
                    honest_exclusion=mean('honest_update_exclusion_rate'),
                    malicious_sample_weight=mean('mean_post_onset_malicious_weight'),
                    malicious_acceptance=mean('malicious_update_acceptance_rate'),
                    runtime_seconds=mean('runtime_seconds_total'),
                    transmitted_bytes=mean('transmitted_bytes'),
                    artifact_bytes=mean('artifact_bytes'),
                    mean_asr=mean('post_onset_mean_asr') if attack == 'model_replacement' else None,
                    endpoint_asr=mean('endpoint_asr') if attack == 'model_replacement' else None))
        for s in SEEDS:
            clean, attack = index.get((d, 'clip_only', s, 'none')), index.get((d, 'clip_only', s, 'model_replacement'))
            matched = bool(clean and attack and condition_pair(clean, attack)['matched'])
            gain = attack['post_onset_mean_asr']-clean['post_onset_mean_clean_trigger_rate'] if matched else None
            effective.append(dict(dataset=d, seed=s, passed=bool(matched and gain >= .10),
                                  asr_excess_over_clean=gain))
    keys = ('experiment_id', 'dataset', 'arm', 'seed', 'attack', 'final_utility', 'post_onset_mean_asr',
            'endpoint_asr', 'post_onset_mean_clean_trigger_rate', 'lower_decile_client_utility',
            'honest_update_exclusion_rate', 'mean_post_onset_malicious_weight',
            'malicious_update_acceptance_rate', 'unsupported_rounds', 'validation_work',
            'runtime_seconds_total', 'transmitted_bytes', 'artifact_bytes')
    return dict(passed=len(records) == 72 and all(row['passed'] for row in rows) and all(r['passed'] for r in effective),
                checks=rows, groups=groups, effective_attack_checks=effective,
                runs=[{k:r[k] for k in keys} for r in records],
                publication_ready=False, independent_confirmation_required=True,
                effective_attack_check_required=True)
