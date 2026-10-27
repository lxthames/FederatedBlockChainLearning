"""Sealed, finite diagnostic followed by a conditional 72-cell development study."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_name, '2')

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from scripts.trust_repair_v16_fixed import study as prior
from scripts.trust_repair_v16_fixed.integrity import IntegrityError
from scripts.trust_repair_v16_fixed.research.models import create_model
from scripts.trust_repair_v16_fixed.runtime import exclusive_lock
from .analysis import DATASETS, SEEDS, calibrate, read_panels, diagnostic_report, end_to_end_report
from .config import ExperimentConfig
from .runner import BenchmarkRunner

ROOT = Path(__file__).resolve().parents[2]
JOB = ROOT/'artifacts/alignment-v17'
OUT = ROOT/'results/alignment-v17'
POLICY = ROOT/'docs/research/ALIGNMENT_V17.md'
NOVELTY = ROOT/'docs/research/ALIGNMENT_V17_NOVELTY.json'
TESTS = ROOT/'tests/test_alignment_v17.py'
LAUNCHER = ROOT/'scripts/run_alignment_v17.ps1'
DATA = prior.DATA
ARMS = ('clip_only', 'current', 'original', 'candidate')
ATTACKS = ('none', 'low_and_slow', 'model_replacement')
read, digest = prior.read, prior.digest


def dump(path, value):
    """Atomic status/report replacement; interrupted readers never see half JSON."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def config(dataset, arm, seed, attack='none', *, passive=False):
    if dataset not in DATASETS or seed not in SEEDS or arm not in ARMS or attack not in ATTACKS:
        raise ValueError('unknown V17 cell')
    if passive and (arm != 'current' or attack == 'low_and_slow'):
        raise ValueError('passive diagnostic only repeats clean/targeted current quorum')
    base = 'clip_only' if arm == 'clip_only' else 'current'
    values = prior.config(dataset, base, seed, attack).to_dict()
    values.update(experiment_id=f"v17-{'passive' if passive else 'development'}-{dataset}-{arm}-{attack}-s{seed}",
                  protocol_version='trust-alignment-v17')
    return ExperimentConfig(**values)


def manifest():
    passive = [dict(role='passive', dataset=d, arm='current', seed=s, attack=t,
                    config=config(d, 'current', s, t, passive=True).to_dict())
               for t in ('none', 'model_replacement') for d in DATASETS for s in SEEDS]
    development = [dict(role='development', dataset=d, arm=a, seed=s, attack=t,
                        config=config(d, a, s, t).to_dict())
                   for t in ATTACKS for d in DATASETS for s in SEEDS for a in ARMS]
    return passive+development


def inputs():
    old = prior.inputs()
    if old != read(prior.JOB/'frozen-inputs.json'):
        raise ValueError('frozen V16 inputs changed')
    return dict(source={str(p.relative_to(ROOT)): digest(p) for p in sorted(Path(__file__).parent.rglob('*.py'))},
        data=old['data'], manifest=manifest(), parent_freeze_sha256=digest(prior.JOB/'frozen-inputs.json'),
        policy_sha256=digest(POLICY), novelty_sha256=digest(NOVELTY), tests_sha256=digest(TESTS),
        launcher_sha256=digest(LAUNCHER), publication_ready=False, automatic_confirmation=False,
        settings=dict(rounds=100, spatial_transforms=147, examples_per_validator=8,
                      panel=5, panels=2, quantile=.995, minimum_recall_gain=.20, maximum_clean_vetoes=3))


def freeze():
    frozen = inputs(); path = JOB/'frozen-inputs.json'
    if path.exists() and read(path) != frozen:
        raise ValueError('V17 was already frozen with different inputs')
    dump(path, frozen)


def sealed_files(folder):
    return {str(p.relative_to(folder)): digest(p) for p in sorted(Path(folder).rglob('*'))
            if p.is_file() and p.name != 'completion.json'}


def trace_hashes(folder):
    trace = read(Path(folder)/'chain-trace.json')
    if [r.get('round', 0) for r in trace] != list(range(101)):
        raise ValueError('incomplete 100-round model trace')
    return [r['model_hash'] for r in trace]


def read_live_status():
    # V16 uses non-atomic status writes; retry only transient status JSON reads.
    for attempt in range(4):
        try:
            return read(prior.JOB/'status.json')
        except json.JSONDecodeError:
            if attempt == 3:
                raise
            time.sleep(.05)


def process_alive(pid):
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                raise OSError('cannot inspect V16 coordinator process')
            return code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False


def validate_passive_parity(folder, reference):
    left, right = read(Path(folder)/'summary.json'), read(Path(reference)/'summary.json')
    ignored = {'protocol_version', 'experiment_id'}
    if {k:v for k,v in left['configuration'].items() if k not in ignored} != {
            k:v for k,v in right['configuration'].items() if k not in ignored}:
        raise ValueError('passive configuration differs beyond version/experiment ID')
    for key in ('training_binding', 'initial_model_hash', 'data_fingerprint', 'test_fingerprint',
                'participants', 'split_bindings', 'local_training_work'):
        if left[key] != right[key]:
            raise ValueError(f'passive parity mismatch: {key}')
    if trace_hashes(folder) != trace_hashes(reference):
        raise ValueError('passive telemetry changed a V16 model hash')
    return dict(passed=True, checked_models=101, reference=str(reference),
                reference_completion_sha256=digest(Path(reference)/'completion.json'),
                declared_changes=sorted(ignored))


def settings_for(row):
    if row['role'] == 'passive':
        ref = prior.OUT/prior.config(row['dataset'], 'current', row['seed'], row['attack']).experiment_id
        return dict(arm='passive', thresholds=None, original_thresholds=None,
                    reference_completion_sha256=digest(ref/'completion.json'),
                    protocol='trust-alignment-v17', decision_feedback=False)
    limits = read(JOB/'calibration.json')['folds'][row['dataset']][str(row['seed'])]['thresholds']
    return dict(arm=row['arm'], thresholds=limits if row['arm'] == 'candidate' else None,
                original_thresholds=read(prior.JOB/'calibration.json')['thresholds'][row['dataset']]
                    if row['arm'] == 'original' else None,
                calibration_sha256=digest(JOB/'calibration.json'),
                qualification_sha256=digest(JOB/'qualification.json'), protocol='trust-alignment-v17')


def verify_cell(folder, c, settings):
    if read(Path(folder)/'completion.json') != sealed_files(folder):
        raise ValueError('V17 completion seal changed')
    summary = read(Path(folder)/'summary.json')
    expected = hashlib.sha256(json.dumps(c.to_dict(), sort_keys=True).encode()).hexdigest()
    if summary['status'] != 'completed' or summary['config_hash'] != expected:
        raise ValueError('V17 config/status mismatch')
    if read(Path(folder)/'alignment-settings.json') != settings:
        raise ValueError('V17 settings changed')
    integrity = read(Path(folder)/'integrity-summary.json')
    if integrity.get('passed') is not True or integrity.get('checked_rounds') != 100:
        raise ValueError('missing V17 integrity checks')
    if summary['diagnostic_input_binding'] != prior.parent.EMPTY_BINDING:
        raise ValueError('unexpected external aggregation input')
    if settings['arm'] == 'passive' and read(Path(folder)/'parity.json').get('passed') is not True:
        raise ValueError('missing passive parity')
    return summary


def audit_v16():
    state = read(prior.JOB/'status.json')
    if state['status'] != 'completed' or state['completed'] != 114 or state['failed'] or state['held']:
        raise ValueError('V16 must complete all 114 cells without failures/holds')
    with exclusive_lock(prior.JOB/'coordinator.lock'):
        if prior.inputs() != read(prior.JOB/'frozen-inputs.json'):
            raise ValueError('V16 frozen source/data mismatch')
        records, seals, pairs = {}, {}, []
        calibration = read(prior.JOB/'calibration.json')
        for row in prior.manifest():
            c = prior.ExperimentConfig(**row['config']); folder = prior.OUT/c.experiment_id
            summary = prior.verify_cell(folder, c, prior.settings_for(row, calibration))
            seals[c.experiment_id] = digest(folder/'completion.json')
            if row['role'] != 'calibration':
                records[row['dataset'], row['arm'], row['seed'], row['attack']] = prior.enrich(summary, folder, row['arm'])
            elif calibration['source_completion_digests'][c.experiment_id] != seals[c.experiment_id]:
                raise ValueError('V16 calibration completion binding mismatch')
        for (dataset, arm, seed, attack), record in records.items():
            if attack != 'none':
                pair = prior.parent.condition_pair(records[dataset, arm, seed, 'none'], record)
                pairs.append(dict(experiment=record['experiment_id'], kind='condition', **pair))
            pair = prior.parent.method_pair(record, records[dataset, 'clip_only', seed, attack])
            pairs.append(dict(experiment=record['experiment_id'], kind='method', **pair))
        for dataset in DATASETS:
            for seed in SEEDS:
                a = prior.OUT/prior.config(dataset, 'clip_only', seed, calibration=True).experiment_id
                b = prior.OUT/prior.config(dataset, 'clip_only', seed).experiment_id
                if trace_hashes(a) != trace_hashes(b):
                    raise ValueError('V16 calibration trajectory differs from clipping control')
        if not all(pair['matched'] for pair in pairs):
            raise ValueError(f"V16 pairing failed: {[p for p in pairs if not p['matched']]}")
        dump(JOB/'v16-audit.json', dict(passed=True, completed_cells=114, completion_digests=seals,
            pairing_checks=pairs, parent_freeze_sha256=digest(prior.JOB/'frozen-inputs.json')))
    return records


def assert_qualification():
    qualification = read(JOB/'qualification.json')
    if not qualification['passed'] or not read(NOVELTY)['supports_bounded_development']:
        raise ValueError('diagnostic/novelty gates do not permit development cells')
    if qualification['calibration_sha256'] != digest(JOB/'calibration.json'):
        raise ValueError('qualified calibration changed')
    if qualification['novelty_sha256'] != digest(NOVELTY):
        raise ValueError('novelty audit changed')
    for experiment, expected in qualification['passive_completion_digests'].items():
        folder = OUT/experiment
        if digest(folder/'completion.json') != expected or read(folder/'completion.json') != sealed_files(folder):
            raise ValueError('qualified passive completion changed')


def execute_cell(index):
    with exclusive_lock(JOB/'cell.lock'), threadpool_limits(limits=2):
        frozen = read(JOB/'frozen-inputs.json')
        if inputs() != frozen:
            raise ValueError('frozen V17 inputs changed')
        row = frozen['manifest'][index]
        if row['role'] == 'development':
            assert_qualification()
        c = ExperimentConfig(**row['config']); folder = OUT/c.experiment_id
        if folder.exists():
            raise FileExistsError('cell refuses an existing output directory')
        settings = settings_for(row)
        dump(JOB/'current-cell.json', dict(pid=os.getpid(), index=index, experiment=c.experiment_id))
        torch.set_num_threads(2)
        runner = BenchmarkRunner(c, folder)
        runner.challenge_arm = settings['arm']
        runner.challenge_thresholds = settings['thresholds']
        runner.original_thresholds = settings['original_thresholds']
        dump(folder/'diagnostic-inputs.json', prior.parent.EMPTY_BINDING)
        dump(folder/'alignment-settings.json', settings)
        try:
            data = DATA/f'{c.dataset}-{c.seed}'
            x, y, ex, ey = [np.load(data/f'{name}.npy', allow_pickle=False) for name in
                            ('features', 'labels', 'development_features', 'development_labels')]
            runner.run_arrays(x, y, lambda: create_model(c.dataset, x.shape[1:], c.num_classes, c.model), (ex, ey))
            if inputs() != frozen:
                raise ValueError('inputs changed during V17 cell')
            integrity = []
            for line in (folder/'events.jsonl').open(encoding='utf-8'):
                event = json.loads(line)
                if event['event_type'] == 'model_integrity':
                    integrity.append(event)
            if [e['round_index'] for e in integrity] != list(range(1, 101)) or not all(
                    e['payload']['passed'] and e['payload']['expected_hash'] == e['payload']['committed_hash'] for e in integrity):
                raise ValueError('incomplete or failed model integrity checks')
            dump(folder/'integrity-summary.json', dict(passed=True, checked_rounds=100,
                 worker_pid=os.getpid(), cpu_arithmetic_reference=True, validated_commit_binding=True))
            if row['role'] == 'passive':
                reference = prior.OUT/prior.config(c.dataset, 'current', c.seed, c.attack).experiment_id
                parity = validate_passive_parity(folder, reference)
                read_panels(folder)
            else:
                parity = dict(applicable=False, reason='end-to-end intervention')
            dump(folder/'parity.json', parity)
            dump(folder/'completion.json', sealed_files(folder))
            verify_cell(folder, c, settings)
        except IntegrityError as error:
            torch.save(error.evidence, folder/'integrity-failure.pt')
            dump(folder/'integrity-failure.json', dict(error=str(error), traceback=traceback.format_exc()))
            raise


def calibration_from_completed():
    panels, bindings = {}, {}
    for dataset in DATASETS:
        for seed in SEEDS:
            c = config(dataset, 'current', seed, passive=True); folder = OUT/c.experiment_id
            verify_cell(folder, c, settings_for(dict(role='passive', dataset=dataset, seed=seed, attack='none')))
            panels[dataset, seed] = read_panels(folder)
            bindings[c.experiment_id] = digest(folder/'completion.json')
    value = dict(calibrate(panels), source_completion_digests=bindings)
    path = JOB/'calibration.json'
    if path.exists() and read(path) != value:
        raise ValueError('frozen clean calibration changed')
    dump(path, value)
    return panels, value


def qualify(clean, calibration, old_records):
    attacked, effective, seals = {}, {}, {}
    for row in manifest()[:12]:
        c = ExperimentConfig(**row['config']); folder = OUT/c.experiment_id
        summary = verify_cell(folder, c, settings_for(row))
        seals[c.experiment_id] = digest(folder/'completion.json')
        if c.attack == 'none':
            continue
        own_clean = read(OUT/config(c.dataset, 'current', c.seed, passive=True).experiment_id/'summary.json')
        own_clean = prior.parent.enrich(own_clean, OUT/config(c.dataset, 'current', c.seed, passive=True).experiment_id)
        attacked_row = prior.parent.enrich(summary, folder)
        if not prior.parent.condition_pair(own_clean, attacked_row)['matched']:
            raise ValueError('passive clean/targeted conditions do not match')
        history = summary['targeted_attack_success_history']
        if len(history) != 100 or not np.isfinite(history).all():
            raise ValueError('incomplete diagnostic ASR history')
        attacked[c.dataset, c.seed] = read_panels(folder), history, c.attack_start_round
        baseline = old_records[c.dataset, 'fedavg', c.seed, 'model_replacement']
        control = old_records[c.dataset, 'fedavg', c.seed, 'none']
        effective[c.dataset, c.seed] = bool(baseline['post_onset_mean_asr']-control['post_onset_mean_clean_trigger_rate'] >= .10)
    report = diagnostic_report(clean, attacked, calibration, effective)
    report.update(calibration_sha256=digest(JOB/'calibration.json'), passive_completion_digests=seals,
                  novelty_sha256=digest(NOVELTY))
    report['diagnostic_passed'] = report['passed']
    report['passed'] = report['passed'] and read(NOVELTY)['supports_bounded_development']
    dump(JOB/'qualification.json', report)
    lines = ['# V17 fixed diagnostic decision', '',
             f"Diagnostic passed: {report['diagnostic_passed']}. Development permitted: {report['passed']}.",
             'Exposed-seed diagnostic; this is not retrained-defense or publication evidence.', '',
             '| Dataset | Seed | Variant | Clean vetoes |', '|---|---:|---|---:|']
    for row in report['clean_checks']:
        lines.append(f"| {row['dataset']} | {row['seed']} | {row['variant']} | {row['vetoes']} |")
    lines += ['', '| FEMNIST seed | Effective attack | High-ASR rounds | Recall gain | Pass |',
              '|---:|---|---:|---:|---|']
    for row in report['primary_checks']:
        gain = 'unavailable' if row['recall_improvement'] is None else f"{100*row['recall_improvement']:.2f}pp"
        lines.append(f"| {row['seed']} | {row['effective_control_attack']} | {row['high_asr_rounds']} | {gain} | {row['passed']} |")
    (JOB/'diagnostic.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return report


def run(wait_for_v16=False):
    with exclusive_lock(JOB/'coordinator.lock'), threadpool_limits(limits=2):
        frozen = read(JOB/'frozen-inputs.json')
        if inputs() != frozen:
            raise ValueError('frozen V17 inputs changed')
        state = dict(status='waiting_for_v16', pid=os.getpid(), completed=0, total=84,
            passive_total=12, conditional_development_total=72, failed=0, held=0,
            started_utc=datetime.now(timezone.utc).isoformat(), publication_ready=False)
        dump(JOB/'status.json', state)
        try:
            while True:
                prior_state = read_live_status()
                status = prior_state['status']
                if status == 'completed':
                    break
                if status != 'running' or not wait_for_v16:
                    raise ValueError(f'V16 prerequisite not complete: {status}')
                if not process_alive(prior_state['pid']):
                    raise RuntimeError('V16 coordinator exited while its status still reports running')
                time.sleep(15)
                if inputs() != frozen:
                    raise ValueError('inputs changed while waiting for V16')
            old_records = audit_v16()
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA required for V17 long experiments')
            dump(JOB/'environment.json', dict(torch=torch.__version__, cuda=torch.version.cuda,
                 gpu=torch.cuda.get_device_name(0), cpu_threads=2))
            records, clean, calibration = [], None, None
            for index, row in enumerate(frozen['manifest']):
                if inputs() != frozen:
                    raise ValueError('frozen V17 inputs changed during experiment')
                if index == 6:
                    clean, calibration = calibration_from_completed()
                if index == 12:
                    qualification = qualify(clean, calibration, old_records)
                    if not qualification['passed']:
                        state.update(status='diagnostic_rejected', blocked_development=72,
                                     reason='fixed diagnostic or novelty gate failed; no threshold search')
                        break
                c = ExperimentConfig(**row['config']); folder = OUT/c.experiment_id
                state.update(status='running', phase=row['role'], current=c.experiment_id,
                             updated_utc=datetime.now(timezone.utc).isoformat())
                dump(JOB/'status.json', state)
                print(f"START {c.experiment_id} ({state['completed']}/84)", flush=True)
                settings = settings_for(row)
                if (folder/'completion.json').exists():
                    summary = verify_cell(folder, c, settings)
                else:
                    if folder.exists():
                        target = folder.with_name(folder.name+f'.interrupted-{time.time_ns()}')
                        if folder.resolve().parent != OUT.resolve() or target.resolve().parent != OUT.resolve():
                            raise ValueError('invalid interrupted output path')
                        folder.rename(target)
                    subprocess.run([sys.executable, '-u', '-m', 'scripts.trust_alignment_v17.study',
                                    '--cell-index', str(index)], cwd=ROOT, check=True,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                    summary = verify_cell(folder, c, settings)
                state['completed'] += 1
                if row['role'] == 'development':
                    records.append(dict(prior.parent.enrich(summary, folder), arm=row['arm'],
                        artifact_bytes=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())))
                    report = end_to_end_report(records, prior.parent.method_pair, prior.parent.condition_pair)
                    dump(JOB/'development-analysis.json', report)
                    write_comparison(report)
                print(f"DONE {c.experiment_id}: utility={summary['final_utility']:.4f}", flush=True)
                dump(JOB/'status.json', state)
            else:
                state.update(status='completed', development_targets_passed=report['passed'])
        except BaseException:
            state.update(status='failed', failed=1, error=traceback.format_exc())
            raise
        finally:
            state['finished_utc'] = datetime.now(timezone.utc).isoformat()
            dump(JOB/'status.json', state)


def write_comparison(report):
    lines = ['# V17 conditional development study', '',
        '100 rounds; exposed seeds 11/17/23. No independent-confirmation claim.', '',
        '| Dataset | Arm | Attack | Accuracy | Mean ASR | Endpoint ASR |', '|---|---|---|---:|---:|---:|']
    fmt = lambda x: 'pending' if x is None else f'{100*x:.2f}%'
    for row in report['groups']:
        lines.append(f"| {row['dataset']} | {row['arm']} | {row['attack']} | {fmt(row['accuracy'])} | {fmt(row['mean_asr'])} | {fmt(row['endpoint_asr'])} |")
    (JOB/'comparison.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--wait-for-v16', action='store_true')
    parser.add_argument('--cell-index', type=int)
    args = parser.parse_args()
    if args.cell_index is not None:
        if not 0 <= args.cell_index < 84:
            parser.error('cell index must be between 0 and 83')
        execute_cell(args.cell_index)
    elif args.freeze:
        freeze(); print('Frozen 12 passive + 72 conditional development cells')
    elif args.run:
        run(args.wait_for_v16)
    else:
        print(json.dumps(manifest(), indent=2))
