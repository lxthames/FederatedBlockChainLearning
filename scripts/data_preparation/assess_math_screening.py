"""Read-only assessment of the bounded mathematical development study.

No models are trained and no protocol is promoted automatically. This report
requires complete matched seed groups and records failed/held experiments.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
JOB = ROOT / 'artifacts/trust-math-lab-v1'
RESULTS = ROOT / 'results/trust-math-lab-v1'
SEEDS = (11, 17, 23)
METHODS = ('paired_loss', 'scaled_loss', 'rank_loss')
POLICY = {
    'version': 'math-screen-assessment-v1',
    'evaluation_role': 'exploratory development, not confirmatory inference',
    'seeds': list(SEEDS),
    'maximum_mean_clean_loss': 0.05,
    'maximum_single_seed_clean_loss': 0.10,
    'minimum_untargeted_damage': 0.005,
    'minimum_targeted_asr_lift': 0.05,
    'minimum_untargeted_defense_benefit': 0.005,
    'minimum_targeted_asr_reduction': 0.05,
    'maximum_mean_attacked_utility_loss_vs_fedavg': 0.05,
    'automatic_promotion': False,
}


def average(values):
    return sum(values) / len(values)


def complete_group(records, variant, attack):
    rows = [r for r in records if r['variant'] == variant and r['attack'] == attack]
    # Duplicate seeds invalidate a group; a dictionary must not hide duplicates.
    if sorted(r['seed'] for r in rows) != list(SEEDS):
        return None
    return {r['seed']: r for r in rows}


def assess(records):
    baseline = complete_group(records, 'fedavg', 'none')
    methods = {}
    for method in METHODS:
        clean = complete_group(records, method, 'none')
        record = {'status': 'incomplete', 'ready_for_reintegration_review': False,
                  'clean': {}, 'attacks': {}}
        methods[method] = record
        if baseline is None or clean is None:
            record['reason'] = 'Complete three-seed clean method and FedAvg groups required.'
            continue
        clean_effects = [clean[s]['final_utility'] - baseline[s]['final_utility'] for s in SEEDS]
        learning = all(baseline[s]['final_utility'] > baseline[s]['development_majority_accuracy'] + .05
                       for s in SEEDS)
        clean_passed = (learning and average(clean_effects) >= -POLICY['maximum_mean_clean_loss']
                        and min(clean_effects) >= -POLICY['maximum_single_seed_clean_loss'])
        record['clean'] = {'passed': clean_passed, 'baseline_learning_passed': learning,
                           'paired_utility_effects': clean_effects}
        any_incomplete = False
        attack_passes = []
        for attack in ('low_and_slow', 'model_replacement'):
            controls = complete_group(records, 'fedavg', attack)
            defended = complete_group(records, method, attack)
            if controls is None:
                record['attacks'][attack] = {'status': 'incomplete_controls'}
                any_incomplete = True
                continue
            if attack == 'low_and_slow':
                effects = [baseline[s]['final_utility'] - controls[s]['final_utility'] for s in SEEDS]
                threshold = POLICY['minimum_untargeted_damage']
            else:
                effects = [controls[s]['mean_targeted_attack_success'] - baseline[s]['mean_clean_trigger_prediction_rate']
                           for s in SEEDS]
                threshold = POLICY['minimum_targeted_asr_lift']
            valid = learning and average(effects) > threshold
            attack_record = {'effective_against_fedavg': valid, 'paired_attack_effects': effects}
            record['attacks'][attack] = attack_record
            if not valid:
                attack_record['status'] = 'ineffective_attack_or_failed_clean_learning'
                attack_passes.append(False)
                continue
            if defended is None:
                attack_record['status'] = 'incomplete_defended_group'
                any_incomplete = True
                continue
            utilities = [defended[s]['final_utility'] - controls[s]['final_utility'] for s in SEEDS]
            if attack == 'low_and_slow':
                benefit = average(utilities) > POLICY['minimum_untargeted_defense_benefit']
            else:
                asr_reductions = [controls[s]['mean_targeted_attack_success'] - defended[s]['mean_targeted_attack_success']
                                  for s in SEEDS]
                attack_record['paired_asr_reductions'] = asr_reductions
                benefit = average(asr_reductions) > POLICY['minimum_targeted_asr_reduction']
            utility_ok = average(utilities) >= -POLICY['maximum_mean_attacked_utility_loss_vs_fedavg']
            attack_record.update(status='assessed', passed=benefit and utility_ok,
                                 paired_attacked_utility_effects=utilities)
            attack_passes.append(benefit and utility_ok)
        ready = clean_passed and not any_incomplete and len(attack_passes) == 2 and all(attack_passes)
        record['ready_for_reintegration_review'] = ready
        record['status'] = ('candidate_for_reintegration_review' if ready else
                            'failed_clean_screen' if not clean_passed else
                            'incomplete' if any_incomplete else 'failed_attack_screen')
    return {'policy': POLICY, 'methods': methods, 'publication_ready': False,
            'next_step': 'Review complete development evidence before a separate recovery/trust reintegration study.'}


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding='utf-8')
    temporary.replace(path)


def report():
    frozen = json.loads((JOB / 'frozen-inputs.json').read_text())
    status = json.loads((JOB / 'status.json').read_text())
    records = []
    invalid = []
    for row in frozen['manifest']:
        config = row['config']
        path = RESULTS / config['experiment_id'] / 'summary.json'
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        expected = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        if result.get('config_hash') != expected or result.get('status') != 'completed':
            invalid.append(str(path.relative_to(ROOT)))
            continue
        # The runner writes summary before the coordinator attaches metadata.
        if 'development_majority_accuracy' not in result or 'variant' not in result:
            continue
        records.append(result)
    value = assess(records)
    value.update(updated_utc=datetime.now(timezone.utc).isoformat(),
                 study_status=status['status'], completed_summaries=len(records),
                 failed_runs=status['failed'], held_runs=status['held'], invalid_summaries=invalid)
    if invalid:
        for result in value['methods'].values():
            result['ready_for_reintegration_review'] = False
            result['status'] = 'invalid_artifact'
    save(JOB / 'assessment.json', value)
    return status['status']


def main(watch=False):
    JOB.mkdir(parents=True, exist_ok=True)
    policy_path = JOB / 'assessment-policy-v1.json'
    policy_record = dict(POLICY, assessor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         timing='Added after the first seed had seven completed clean runs; development only.')
    if policy_path.exists() and json.loads(policy_path.read_text()) != policy_record:
        raise RuntimeError('Assessment policy changed; use a new assessment version.')
    save(policy_path, policy_record)
    while True:
        state = report()
        if not watch or state not in {'starting', 'running'}:
            break
        time.sleep(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    main(parser.parse_args().watch)
