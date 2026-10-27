"""Post-hoc explanatory audit. Never changes V17 thresholds or qualification.

Attack labels are used only for offline interpretation of archived telemetry.
No model is trained and no defense decision is changed.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.trust_alignment_v17.analysis import read_panels, veto_rounds
from scripts.trust_alignment_v17.behavior import STAGES


def main():
    root = Path(__file__).resolve().parents[1]
    bindings = {}

    def read_json(path):
        raw = path.read_bytes()
        bindings[str(path.relative_to(root))] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    calibration = read_json(root / 'artifacts/alignment-v17/calibration.json')
    qualification = read_json(root / 'artifacts/alignment-v17/qualification.json')
    rows = []
    for seed in (11, 17, 23):
        folder = root / f'results/alignment-v17/v17-passive-femnist-current-model_replacement-s{seed}'
        summary = read_json(folder / 'summary.json')
        clean_folder = folder.with_name(folder.name.replace('model_replacement', 'none'))
        clean = read_json(clean_folder / 'summary.json')
        seal = read_json(folder / 'completion.json')
        events = folder / 'events.jsonl'
        bindings[str(events.relative_to(root))] = hashlib.sha256(events.read_bytes()).hexdigest()
        panels = read_panels(folder)
        thresholds = calibration['folds']['femnist'][str(seed)]['thresholds']
        onset = summary['configuration']['attack_start_round']
        asr = summary['targeted_attack_success_history']
        target = summary['configuration']['backdoor_target_label']
        variants = {}
        for variant in ('probability_independent', 'label_aligned'):
            vetoes = veto_rounds(panels, thresholds, variant)
            signals = {}
            for metric in ('absolute', 'incremental', 'clean_excess'):
                hit_rounds = []
                for rnd in range(onset, 101):
                    def exceeds(stage):
                        panel = panels[rnd, stage]
                        limit = thresholds[stage][variant][metric]
                        if metric == 'clean_excess':
                            return sum(x > limit for x in panel[metric]) >= 2
                        return panel['scores'][variant][metric] > limit
                    if any(exceeds(stage) for stage in STAGES):
                        hit_rounds.append(rnd)
                signals[metric] = hit_rounds
            variants[variant] = dict(
                first_alarm=next((r for r in range(onset, 101) if vetoes[r]), None),
                signal_alarm_rounds=signals,
            )
        observations = []
        for rnd in (11, 30, 100):
            clean_counts, clean_n = 0, 0
            maximum_patch_fractions = []
            paths = sorted((folder / 'behavior').glob(f'r{rnd:03}-*.npz'))
            if len(paths) != 10:
                raise ValueError('Expected both complete panels')
            for path in paths:
                bindings[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
                with np.load(path) as archive:
                    counts = archive['candidate_class_counts']
                    n = int(archive['sample_count'])
                    clean_counts += int(counts[0, target])
                    clean_n += n
                    maximum_patch_fractions.append(float(counts[1:148, target].max() / n))
            observations.append(dict(
                round=rnd, asr=asr[rnd-1], accuracy=summary['accuracy_history'][rnd-1],
                matched_clean_accuracy=clean['accuracy_history'][rnd-1],
                matched_clean_trigger_rate=clean['clean_trigger_prediction_history'][rnd-1],
                private_clean_target_count=clean_counts, private_sample_count=clean_n,
                mean_validator_max_patch_target_fraction=float(np.mean(maximum_patch_fractions)),
                label_absolute_scores={st:panels[rnd, st]['scores']['label_aligned']['absolute'] for st in STAGES},
            ))
        reference = next(r for r in qualification['attack_checks']
                         if r['dataset'] == 'femnist' and r['seed'] == seed
                         and r['variant'] == 'probability_independent')
        rows.append(dict(seed=seed, onset=onset,
                         first_asr_over_half=next(r for r in range(onset, 101) if asr[r-1] > .5),
                         maximum_possible_recall_gain=1-reference['high_asr_recall'],
                         variants=variants, observations=observations))
    output = root / 'artifacts/alignment-v17-followup'
    output.mkdir(exist_ok=True)
    report = dict(
        analysis='post-hoc explanation only; original failed qualification remains unchanged',
        limitations=[
            'Passive trajectories do not estimate retrained defense performance.',
            'Early high ASR includes broad class collapse; timing alone does not establish trigger-specific implantation.',
            'Validator patch maxima are descriptive and may refer to different transformations.',
            'These are exposed development seeds, not independent confirmation.',
        ], femnist=rows, input_sha256=bindings,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output / 'timing-audit.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    lines = [
        '# V17 post-hoc timing and collapse audit', '',
        'Exploratory interpretation of existing telemetry. The frozen diagnostic remains rejected; no thresholds or decisions were changed.', '',
        '| FEMNIST seed | First ASR >50% | Probability first alarm | Candidate first alarm | Candidate incremental alarms | Maximum possible recall gain |',
        '|---:|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        variants = row['variants']
        lines.append(f"| {row['seed']} | {row['first_asr_over_half']} | {variants['probability_independent']['first_alarm']} | {variants['label_aligned']['first_alarm']} | {len(variants['label_aligned']['signal_alarm_rounds']['incremental'])} | {100*row['maximum_possible_recall_gain']:.2f}pp |")
    lines += ['',
        'At round 11 in each seed, all 80 sampled private examples are classified as the target on clean inputs and on every spatial transformation. The label absolute contrast is zero in both panels. Task accuracy is only 4.5-5.3%. This is observed broad prediction collapse, so 100% ASR at this point is insufficient evidence of trigger-specific behavior.', '',
        'By round 100, task accuracy is 75.2-80.7% while ASR is 99.9-100%; the matched clean runs have trigger target rates near 1.6-1.8%. Thus early collapse does not explain away the sustained backdoor failure.', '',
        'The reference recall leaves only 6.82-11.24 percentage points of improvement available. The preregistered +20pp criterion is unattainable at the observed reference scores. This design limitation does not rescue V17: its measured gains are negative or zero. Do not relax the gate retrospectively.', '',
        'Next causal question: does the same blind spot persist when attacks begin from a model with useful clean performance? A separately frozen onset comparison should hold attack exposure and decision rules fixed, report clean target rates alongside ASR, and measure utility throughout. It must not be presented as a validated new defense.', '',
        'Detailed values and input hashes: timing-audit.json. Target labels are used only in this offline audit.',
    ]
    # Verify the stronger all-transform collapse statement used in the prose.
    for row in rows:
        folder = root / f"results/alignment-v17/v17-passive-femnist-current-model_replacement-s{row['seed']}"
        for path in (folder / 'behavior').glob('r011-*.npz'):
            with np.load(path) as archive:
                assert np.all(archive['candidate_class_counts'][:148, 0] == int(archive['sample_count']))
    (output / 'timing-audit.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
