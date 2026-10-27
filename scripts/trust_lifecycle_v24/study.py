"""Frozen lifecycle study: repair, admission, and attack persistence."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_name, '2')

import argparse
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from scripts.trust_periodic_v22 import study as v22
from scripts.trust_periodic_v22.online import OnlineRepair, DEPLOYMENT, select_committed
from scripts.trust_anchor_v20 import study as v20
from scripts.trust_repair_v16_fixed.integrity import snapshot, state_hash, assert_hash
from scripts.trust_repair_v16_fixed.research.models import create_model
from scripts.trust_repair_v16_fixed.research.splits import make_splits
from scripts.trust_repair_v16_fixed.runtime import exclusive_lock
from .runner import BenchmarkRunner

ROOT = v22.ROOT
JOB = ROOT / 'artifacts/lifecycle-v24'
OUT = ROOT / 'results/lifecycle-v24'
POLICY_FILE = ROOT / 'docs/research/LIFECYCLE_V24.md'
LAUNCHER = ROOT / 'scripts/run_lifecycle_v24.ps1'
SEEDS = (11, 17, 23)
CONDITIONS = ('clean', 'attack_stop', 'continuous')
ARMS = ('quorum', 'periodic', 'forced')

read, dump, digest = v22.read, v22.dump, v22.digest


def config(seed, condition, arm):
    if seed not in SEEDS or condition not in CONDITIONS or arm not in ARMS:
        raise ValueError('unknown lifecycle cell')
    attack = 'none' if condition == 'clean' else 'model_replacement'
    base = v22.config(seed, attack, 'current')
    return replace(base, experiment_id=f'v24-femnist-{arm}-{condition}-s{seed}')


def stop_round(condition):
    return 0 if condition == 'clean' else (50 if condition == 'attack_stop' else 100)


def manifest():
    return [dict(condition=c, seed=s, arm=a, id=config(s, c, a).experiment_id)
            for c in CONDITIONS for s in SEEDS for a in ARMS]


class ForcedAdmissionRepair(OnlineRepair):
    """Diagnostic arm that commits the candidate even when private verification fails."""
    def __init__(self, deployment=DEPLOYMENT, repair=None, anchor=None):
        super().__init__('v20_periodic', deployment, repair=repair, anchor=anchor)
        self.variant = 'forced'

    def attempt(self, source, x, y, split, rnd, contributors, seed):
        outcome = super().attempt(source, x, y, split, rnd, contributors, seed)
        decision = deepcopy(outcome.payload['decision'])
        verifier_passed = bool(decision['passed'])
        decision['base_verifier_passed'] = verifier_passed
        decision['forced_admission'] = True
        decision['passed'] = True
        outcome.committed = select_committed(source, outcome.candidate, decision)
        outcome.payload.update(
            arm='forced', decision=decision, held=False,
            committed_model_hash=state_hash(snapshot(outcome.committed)),
        )
        return outcome


class ProgressLogger:
    def __init__(self, logger, folder, cell_id):
        self.logger, self.folder, self.cell_id = logger, folder, cell_id
        self.attempts = self.admitted = 0

    def log(self, event_type, round_index, payload):
        self.logger.log(event_type, round_index, payload)
        if event_type == 'online_repair':
            self.attempts += 1
            self.admitted += int(payload['decision']['passed'])
        if event_type == 'round_completed':
            dump(self.folder / 'progress.json', dict(
                cell=self.cell_id, round=round_index, total_rounds=100,
                accuracy=payload['accuracy'], repair_attempts=self.attempts,
                repair_admissions=self.admitted,
                repair_holds=self.attempts - self.admitted,
                runtime_seconds=payload['runtime_seconds'],
                updated_utc=datetime.now(timezone.utc).isoformat()))
            if round_index == 1 or round_index % 10 == 0:
                print(f'ROUND {round_index}/100 {self.cell_id}; '
                      f'repairs {self.admitted}/{self.attempts}', flush=True)


class OfflineObserver:
    """Scores source, candidate, and committed models after each repair decision."""
    def __init__(self, features, labels, target, clients):
        self.features, self.labels, self.target, self.clients = features, labels, target, clients
        self.records = []

    def __call__(self, outcome, rnd):
        started = time.perf_counter()
        source = v22.parent.evaluate(outcome.source, self.features, self.labels, self.target, self.clients)
        candidate = v22.parent.evaluate(outcome.candidate, self.features, self.labels, self.target, self.clients)
        committed = candidate if outcome.payload['decision']['passed'] else source
        assert_hash(outcome.committed, committed['model_hash'], 'offline committed equality')
        self.records.append(dict(round=rnd, source=source, candidate=candidate,
                                 committed=committed,
                                 admitted=outcome.payload['decision']['passed'],
                                 seconds=time.perf_counter() - started))


def inputs():
    v20.assert_frozen()
    v20_state = read(v20.JOB / 'status.json')
    if v20_state['status'] != 'checkpoint_promising' or not read(v20.JOB / 'audit.json')['passed']:
        raise ValueError('V20 checkpoint evidence is incomplete')
    source_files = [Path(__file__).parent / name for name in ('__init__.py', 'runner.py', 'study.py')]
    return dict(
        source={str(p.relative_to(ROOT)): digest(p) for p in source_files},
        v20_freeze_sha256=digest(v20.JOB / 'frozen-inputs.json'),
        v20_analysis_sha256=digest(v20.JOB / 'analysis.json'),
        v20_audit_sha256=digest(v20.JOB / 'audit.json'),
        manifest=[dict(r, config=config(r['seed'], r['condition'], r['arm']).to_dict(),
                       attack_stop_round=stop_round(r['condition'])) for r in manifest()],
        repair_policy=vars(v20.POLICY), anchor_policy=vars(v20.ANCHOR),
        deployment=vars(DEPLOYMENT), policy_sha256=digest(POLICY_FILE),
        launcher_sha256=digest(LAUNCHER), automatic_followup=False,
        publication_ready=False)


def assert_frozen():
    expected = json.loads(json.dumps(inputs()))
    if expected != read(JOB / 'frozen-inputs.json'):
        raise ValueError('lifecycle-v24 frozen inputs changed')


def trace_hashes(folder):
    return [r['model_hash'] for r in read(folder / 'chain-trace.json')]


def extract_row(summary, row, online_rows, offline, integrity, events):
    targeted = row['condition'] != 'clean'
    history = summary['targeted_attack_success_history']
    if targeted and len(history) != 100:
        raise ValueError('incomplete targeted trajectory')
    onset = 11
    stop = stop_round(row['condition'])
    attempted = len(online_rows)
    admitted = sum(int(r['decision']['passed']) for r in online_rows)
    result = dict(
        seed=row['seed'], condition=row['condition'], attack=summary['configuration']['attack'], arm=row['arm'],
        final_accuracy=summary['final_accuracy'],
        mean_post_onset_asr=float(np.mean(history[onset - 1:])) if targeted else None,
        endpoint_asr=history[-1] if targeted else None,
        mean_during_attack=float(np.mean(history[onset - 1:stop])) if targeted else None,
        mean_after_attack_stop=float(np.mean(history[stop:])) if row['condition'] == 'attack_stop' else None,
        rounds_at_least_90=sum(x >= .9 for x in history[onset - 1:]) if targeted else None,
        lower_decile_client_utility=summary['lower_decile_client_utility'],
        repair_attempts=attempted, repair_admissions=admitted, repair_holds=attempted - admitted,
        repair_seconds=sum(r['seconds'] for r in online_rows),
        local_training_work=summary['local_training_work'], validation_work=summary['validation_work'],
        extra_repair_example_presentations=sum(r['work'].get('training_examples', 0) +
                                               r['work'].get('proposal_examples', 0) +
                                               r['work'].get('diagnostic_examples', 0) for r in online_rows),
        extra_teacher_presentations=sum(r['work'].get('teacher_examples', 0) for r in online_rows),
        extra_panel_presentations=sum(r['verification_presentations'] for r in online_rows),
        extra_repair_transmitted_bytes=sum(r['extra_transmitted_bytes'] for r in online_rows),
        transmitted_bytes=summary['transmitted_bytes'], runtime_seconds=summary['runtime_seconds_total'],
        accepted_malicious_weight_history=summary['accepted_malicious_weight_history'],
        mean_post_onset_malicious_weight=summary['mean_post_onset_malicious_weight'],
        malicious_update_acceptance_rate=summary['malicious_update_acceptance_rate'],
        integrity=integrity, reused=False)
    return result


def execute(index):
    with exclusive_lock(JOB / 'cell.lock'), threadpool_limits(limits=2):
        assert_frozen()
        row = manifest()[index]
        cfg = config(row['seed'], row['condition'], row['arm'])
        folder = OUT / row['id']
        if folder.exists():
            raise FileExistsError(f'refusing existing output: {folder}')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for lifecycle-v24')
        torch.set_num_threads(2)
        x, y, tx, ty = v22.parent.load_arrays(row['seed'])
        split = make_splits(y, cfg, external_test=True)
        runner = BenchmarkRunner(cfg, folder)
        runner.attack_stop_round = stop_round(row['condition']) or cfg.rounds
        runner.challenge_arm = 'current'
        runner.logger = ProgressLogger(runner.logger, folder, row['id'])
        observer = OfflineObserver(tx, ty, cfg.backdoor_target_label, split.natural_test_clients)
        if row['arm'] == 'periodic':
            runner.online_controller = OnlineRepair('v20_periodic', DEPLOYMENT, v20.POLICY, v20.ANCHOR)
            runner.offline_observer = observer
        elif row['arm'] == 'forced':
            runner.online_controller = ForcedAdmissionRepair(DEPLOYMENT, v20.POLICY, v20.ANCHOR)
            runner.offline_observer = observer
        runner.run_arrays(x, y,
            lambda: create_model(cfg.dataset, x.shape[1:], cfg.num_classes, cfg.model),
            (tx, ty))
        trace = trace_hashes(folder)
        final_state = snapshot(runner.final_model)
        final_hash = state_hash(final_state)
        if final_hash != trace[-1]:
            raise ValueError('final state differs from committed trace')
        torch.save(dict(model_hash=final_hash, state=final_state), folder / 'final.pt')
        events = [json.loads(line) for line in (folder / 'events.jsonl').open(encoding='utf-8')]
        integrity_events = [e for e in events if e['event_type'] == 'model_integrity']
        if len(integrity_events) != 100 or not all(e['payload']['passed'] for e in integrity_events):
            raise ValueError('global model integrity checks incomplete')
        online = [e['payload'] for e in events if e['event_type'] == 'online_repair']
        expected_repairs = [25, 50, 75, 100] if row['arm'] in ('periodic', 'forced') else []
        if [r['round'] for r in online] != expected_repairs:
            raise ValueError('repair schedule mismatch')
        for r in online:
            if r['committed_model_hash'] != trace[r['round']]:
                raise ValueError('repair selection differs from committed trace')
        dump(folder / 'offline-repair-evaluation.json', dict(records=observer.records))
        audit = dict(passed=True, arm=row['arm'], condition=row['condition'], checked_rounds=100,
                     attack_stop_round=stop_round(row['condition']), online_decisions=len(online),
                     final_model_hash=final_hash)
        dump(folder / 'integrity-summary.json', audit)
        summary = extract_row(read(folder / 'summary.json'), row, online, observer.records, audit, events)
        final_eval = v22.parent.evaluate(runner.final_model, tx, ty, cfg.backdoor_target_label,
                                         split.natural_test_clients)
        if final_eval['model_hash'] != final_hash:
            raise ValueError('final evaluation hash mismatch')
        dump(folder / 'final-evaluation.json', final_eval)
        summary['final_evaluation'] = final_eval
        dump(folder / 'online-summary.json', summary)
        dump(folder / 'settings.json', dict(row=row, attack_stop_round=stop_round(row['condition']),
                                            freeze_sha256=digest(JOB / 'frozen-inputs.json')))
        assert_frozen()
        dump(folder / 'completion.json', v22.parent.seal_files(folder))
        v22.parent.verify_seal(folder)


def collect_rows():
    rows = []
    for row in manifest():
        folder = OUT / row['id']
        if (folder / 'completion.json').exists():
            v22.parent.verify_seal(folder)
            rows.append(read(folder / 'online-summary.json'))
    return rows


def report():
    rows = collect_rows()
    rows.sort(key=lambda r: (CONDITIONS.index(r['condition']), r['seed'], ARMS.index(r['arm'])))
    dump(JOB / 'results.json', dict(records=rows, completed=len(rows), total=len(manifest()),
                                    status='complete' if len(rows) == len(manifest()) else 'running'))
    lines = ['# V24 lifecycle continuation study', '',
             f'Status: {"complete" if len(rows) == len(manifest()) else "running"}; cells: {len(rows)}/{len(manifest())}.',
             'FEMNIST seeds 11/17/23; repairs at rounds 25, 50, 75, 100.',
             'Attack-stop condition poisons through round 50, then uses identical training without poisoning.', '',
             '| Condition | Seed | Arm | Final accuracy | Mean post-onset ASR | Mean during attack | Mean after stop | Endpoint ASR | Repairs admitted |',
             '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        def pct(value): return 'n/a' if value is None else f'{100 * value:.2f}%'
        lines.append(f"| {r['condition']} | {r['seed']} | {r['arm']} | {100*r['final_accuracy']:.2f}% | "
                     f"{pct(r['mean_post_onset_asr'])} | {pct(r['mean_during_attack'])} | "
                     f"{pct(r['mean_after_attack_stop'])} | {pct(r['endpoint_asr'])} | "
                     f"{r['repair_admissions']}/{r['repair_attempts']} |")
    lines += ['', 'All ASR values use the complete committed trajectory; endpoint round 100 is phase-sensitive.',
              'Forced admission is a diagnostic arm for separating repair quality from private admission.',
              'This study uses exposed development data and an in-bank trigger; it is not independent confirmation.']
    (JOB / 'comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return rows


def run():
    with exclusive_lock(JOB / 'coordinator.lock'):
        assert_frozen()
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for lifecycle-v24')
        state = dict(status='running', pid=os.getpid(), completed=0, total=len(manifest()), failed=0,
                     started_utc=datetime.now(timezone.utc).isoformat(), publication_ready=False)
        dump(JOB / 'status.json', state)
        dump(JOB / 'environment.json', dict(torch=torch.__version__, cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0)))
        try:
            for i, row in enumerate(manifest()):
                assert_frozen()
                state.update(current=row['id'], updated_utc=datetime.now(timezone.utc).isoformat())
                dump(JOB / 'status.json', state)
                folder = OUT / row['id']
                if not (folder / 'completion.json').exists():
                    if folder.exists():
                        folder.rename(folder.with_name(folder.name + f'.interrupted-{time.time_ns()}'))
                    print(f'START {i+1}/{len(manifest())} {row["id"]}', flush=True)
                    result = subprocess.run([sys.executable, '-u', '-m',
                        'scripts.trust_lifecycle_v24.study', '--cell', str(i)], cwd=ROOT)
                    if result.returncode:
                        raise RuntimeError(f'worker failed: {row["id"]} exit {result.returncode}')
                v22.parent.verify_seal(folder)
                state['completed'] = len(collect_rows())
                report()
                dump(JOB / 'status.json', state)
                print(f'DONE {i+1}/{len(manifest())} {row["id"]}', flush=True)
            state.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat())
            dump(JOB / 'status.json', state)
        except Exception as error:
            state.update(status='failed', failed=1, error=str(error), traceback=traceback.format_exc())
            dump(JOB / 'status.json', state)
            raise


def main():
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--freeze', action='store_true')
    action.add_argument('--run', action='store_true')
    action.add_argument('--report', action='store_true')
    action.add_argument('--cell', type=int)
    args = parser.parse_args()
    if args.freeze:
        value = json.loads(json.dumps(inputs()))
        path = JOB / 'frozen-inputs.json'
        if path.exists() and read(path) != value:
            raise ValueError('lifecycle-v24 already frozen differently')
        dump(path, value)
        print(f'Frozen lifecycle-v24: {digest(path)}')
    elif args.run:
        run()
    elif args.report:
        assert_frozen()
        report()
    else:
        execute(args.cell)


if __name__ == '__main__':
    main()
