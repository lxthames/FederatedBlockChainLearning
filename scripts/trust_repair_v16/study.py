"""Frozen 100-round development study of aggregate validation and omission repair."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(name, '2')
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback
import subprocess
import sys
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from scripts.trust_security_v13 import study as parent
from scripts.trust_challenge_v14 import study as v14
from scripts.trust_repair_v15 import study as v15
from .integrity import IntegrityError
from .research.config import ExperimentConfig
from .research.models import create_model
from .research.runner_v2 import BenchmarkRunner
from .runtime import exclusive_lock
from .repair import ARMS, QUANTILE, CAP, PANEL, GRID, EPS

ROOT = Path(__file__).resolve().parents[2]
DATA = parent.DATA
JOB = ROOT/'artifacts/repair-v16'
OUT = ROOT/'results/repair-v16'
POLICY = ROOT/'docs/research/REPAIR_V16.md'
SEEDS = (11, 17, 23)
DATASETS = ('fashion_mnist', 'femnist')
ATTACKS = ('none', 'low_and_slow', 'model_replacement')
ROUNDS = 100
FILES = parent.FILES+('repair-settings.json','integrity-summary.json')
read, dump, digest = parent.read, parent.dump, parent.digest


def config(dataset, arm, seed, attack='none', *, calibration=False):
    if dataset not in DATASETS or arm not in ARMS or seed not in SEEDS or attack not in ATTACKS:
        raise ValueError('unknown V16 cell')
    if calibration and (arm != 'clip_only' or attack != 'none'):
        raise ValueError('calibration must be clean clipping')
    base = arm if arm in ('fedavg','clip_only','fedval') else parent.CANDIDATE if arm == 'current' else 'clip_only'
    values = parent.config(dataset, base, seed, attack).to_dict()
    role = 'calibration' if calibration else 'evaluation'
    values.update(experiment_id=f'v16-{role}-{dataset}-{arm}-{attack}-s{seed}',
                  rounds=ROUNDS, protocol_version='trust-repair-v16')
    return ExperimentConfig(**values)


def manifest():
    calibration = [dict(dataset=d, arm='clip_only', seed=s, attack='none', role='calibration',
        config=config(d,'clip_only',s,calibration=True).to_dict()) for d in DATASETS for s in SEEDS]
    evaluation = [dict(dataset=d, arm=a, seed=s, attack=t, role='evaluation', config=config(d,a,s,t).to_dict())
        for t in ATTACKS for d in DATASETS for s in SEEDS for a in ARMS]
    return calibration+evaluation


def inputs():
    if parent.inputs() != read(parent.JOB/'frozen-inputs.json'):
        raise ValueError('V13 source/data changed')
    if v14.inputs() != read(v14.JOB/'frozen-inputs.json'):
        raise ValueError('V14 source/data changed')
    if v15.inputs() != read(v15.JOB/'frozen-inputs.json'):
        raise ValueError('V15 source/data changed')
    return dict(source={str(p.relative_to(ROOT)):digest(p) for p in sorted(Path(__file__).parent.rglob('*.py'))},
        data=read(parent.JOB/'frozen-inputs.json')['data'], manifest=manifest(),
        parent_freeze_sha256=digest(v14.JOB/'frozen-inputs.json'), policy_sha256=digest(POLICY),
        v15_freeze_sha256=digest(v15.JOB/'frozen-inputs.json'),
        integrity_audit_sha256=digest(ROOT/'docs/research/REPAIR_V15_INTEGRITY_AUDIT.md'),
        tests_sha256=digest(ROOT/'tests/test_repair_v16.py'),
        settings=dict(rounds=ROUNDS,cap=CAP,panel=PANEL,grid=GRID,quantile=QUANTILE,improvement=EPS),
        evaluation_role='long-horizon development on exposed seeds; not independent confirmation',
        publication_ready=False, automatic_confirmation=False)


def freeze():
    value = inputs()
    path = JOB/'frozen-inputs.json'
    if path.exists() and read(path) != value:
        raise ValueError('frozen V16 inputs changed')
    dump(path,value)
    return value


def sealed_files(folder):
    paths = [folder/n for n in FILES]
    paths += sorted((folder/'behavior').glob('*.npz'))+sorted((folder/'checkpoints').glob('*.pt'))
    return {str(p.relative_to(folder)):digest(p) for p in paths}


def verify_cell(folder,c,settings):
    if read(folder/'completion.json') != sealed_files(folder):
        raise ValueError('V16 completion seal changed')
    summary = read(folder/'summary.json')
    expected = hashlib.sha256(json.dumps(c.to_dict(),sort_keys=True).encode()).hexdigest()
    if summary['status'] != 'completed' or summary['config_hash'] != expected:
        raise ValueError('config/status mismatch')
    if read(folder/'repair-settings.json') != settings:
        raise ValueError('repair/calibration binding changed')
    if read(folder/'diagnostic-inputs.json') != parent.EMPTY_BINDING:
        raise ValueError('unexpected diagnostic intervention')
    integrity=read(folder/'integrity-summary.json')
    if integrity.get('passed') is not True or integrity.get('checked_rounds')!=c.rounds:
        raise ValueError('incomplete runtime integrity checks')
    return summary


def settings_for(row,calibration):
    passive = row['role'] == 'calibration'
    return dict(arm='calibrate' if passive else row['arm'],
        thresholds=None if passive or row['arm'] not in ('aggregate','repair') else calibration['thresholds'][row['dataset']],
        calibration_sha256=None if passive else digest(JOB/'calibration.json'),
        cap=CAP,panel=PANEL,grid=GRID,maximum_omissions=1)


def calibrate():
    thresholds,bindings = {},{}
    for dataset in DATASETS:
        values = {stage:{k:[] for k in ('absolute','incremental','clean_excess')} for stage in ('screen','verify')}
        for seed in SEEDS:
            c = config(dataset,'clip_only',seed,calibration=True)
            folder = OUT/c.experiment_id
            verify_cell(folder,c,settings_for(dict(role='calibration',arm='clip_only',dataset=dataset),None))
            bindings[c.experiment_id] = digest(folder/'completion.json')
            with (folder/'events.jsonl').open(encoding='utf-8') as handle:
                for line in handle:
                    event = json.loads(line)
                    if event['event_type'] == 'repair_vote':
                        p = event['payload']
                        for key,value in p['metrics'].items():
                            values[p['stage']][key].append(value)
        if any(not vals for stage in values.values() for vals in stage.values()):
            raise ValueError('incomplete clean calibration')
        thresholds[dataset] = {stage:{key:float(np.quantile(vals,QUANTILE,method='higher'))
            for key,vals in metrics.items()} for stage,metrics in values.items()}
    value = dict(thresholds=thresholds,source_completion_digests=bindings,quantile=QUANTILE,
        source='private clean clipping trajectories only; empirical quantiles; no FPR/security guarantee')
    path = JOB/'calibration.json'
    if path.exists() and read(path) != value:
        raise ValueError('sealed calibration changed')
    dump(path,value)
    return value


def find(records,dataset,arm,seed,attack='none'):
    return next((r for r in records if (r['dataset'],r['arm'],r['seed'],r['attack']) == (dataset,arm,seed,attack)),None)


def gate(records,arm):
    cells = []
    for dataset in DATASETS:
        for seed in SEEDS:
            row,base = find(records,dataset,arm,seed),find(records,dataset,'clip_only',seed)
            pair = parent.method_pair(row,base) if row and base else dict(matched=False,mismatches=['missing clean cell'])
            passed = bool(row and base and pair['matched'] and base['final_utility'] >= .2
                and row['final_utility'] >= base['final_utility']-.05 and row['unsupported_rounds'] <= 3
                and row['honest_validator_penalty_amount'] == 0)
            cells.append(dict(dataset=dataset,seed=seed,passed=passed,pairing=pair))
    return dict(passed=all(c['passed'] for c in cells),cells=cells)


def enrich(summary,folder,arm):
    result = dict(parent.enrich(summary,folder),arm=arm)
    counts = dict(aggregate_accepts=0,repair_attempts=0,successful_repairs=0,aggregate_holds=0,omission_screens=0)
    search_rounds = set()
    with (folder/'events.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            event = json.loads(line)
            p = event['payload']
            if event['event_type'] == 'repair_decision':
                if p['action'] == 'repair': counts['successful_repairs'] += 1
                elif p['action'] == 'hold': counts['aggregate_holds'] += 1
                elif p['action'] == 'accept': counts['aggregate_accepts'] += 1
            if event['event_type'] == 'repair_panel' and p['stage']=='screen' and p['omitted'] is not None:
                counts['omission_screens'] += 1
                search_rounds.add(event['round_index'])
    counts['repair_attempts']=len(search_rounds)
    result.update(counts)
    result['transmitted_bytes']=summary['transmitted_bytes']
    result['lower_decile_client_utility']=summary['lower_decile_client_utility']
    result['worst_client_utility']=summary['worst_client_utility']
    result['artifact_bytes']=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())
    return result


def analyze(records,state):
    groups,pairs = [],[]
    for row in records:
        if row['attack'] != 'none':
            clean = find(records,row['dataset'],row['arm'],row['seed'])
            pairs.append(dict(experiment=row['experiment_id'],**parent.condition_pair(clean,row)))
    for dataset in DATASETS:
        for arm in ARMS:
            for attack in ATTACKS:
                rows = [r for r in records if (r['dataset'],r['arm'],r['attack']) == (dataset,arm,attack)]
                complete = sorted(r['seed'] for r in rows) == list(SEEDS)
                if attack != 'none':
                    complete = complete and all(next(p for p in pairs if p['experiment']==r['experiment_id'])['matched'] for r in rows)
                avg = lambda key:float(np.mean([r[key] for r in rows])) if complete else None
                groups.append(dict(dataset=dataset,arm=arm,attack=attack,complete=complete,
                    accuracy=avg('final_utility'),mean_asr=avg('post_onset_mean_asr') if attack=='model_replacement' else None,
                    endpoint_asr=avg('endpoint_asr') if attack=='model_replacement' else None,
                    honest_exclusion=avg('honest_update_exclusion_rate'),unsupported=avg('unsupported_rounds'),
                    runtime_seconds=avg('runtime_seconds_total'),
                    lower_decile_utility=avg('lower_decile_client_utility'),
                    successful_repairs=avg('successful_repairs'),aggregate_holds=avg('aggregate_holds'),
                    transmitted_bytes=avg('transmitted_bytes'),artifact_bytes=avg('artifact_bytes')))
    keys = ('experiment_id','dataset','arm','seed','attack','final_utility','post_onset_mean_asr','endpoint_asr',
        'honest_update_exclusion_rate','malicious_update_acceptance_rate','unsupported_rounds','validation_work','runtime_seconds_total',
        'post_onset_mean_clean_trigger_rate','lower_decile_client_utility','worst_client_utility',
        'repair_attempts','successful_repairs','aggregate_holds','omission_screens','transmitted_bytes','artifact_bytes')
    compact = [{k:r[k] for k in keys} for r in records]
    dump(JOB/'analysis.json',dict(status=state,groups=groups,runs=compact,condition_pairs=pairs,
        gates={a:gate(records,a) for a in ('current','aggregate','repair')},publication_ready=False,automatic_promotion=False))
    lines = ['# V16 long-horizon development study','','100 rounds; exposed seeds 11/17/23. Held attacks are not defense successes.','',
        '| Dataset | Arm | Attack | Accuracy | Mean ASR | Endpoint ASR |','|---|---|---|---:|---:|---:|']
    fmt = lambda x:'pending/held' if x is None else f'{100*x:.2f}%'
    for g in groups:
        lines.append(f"| {g['dataset']} | {g['arm']} | {g['attack']} | {fmt(g['accuracy'])} | {fmt(g['mean_asr'])} | {fmt(g['endpoint_asr'])} |")
    (JOB/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def execute_cell(index):
    with exclusive_lock(JOB/'cell.lock'),threadpool_limits(limits=2):
        frozen=read(JOB/'frozen-inputs.json')
        if inputs()!=frozen: raise ValueError('child frozen inputs changed')
        row=frozen['manifest'][index]
        c=ExperimentConfig(**row['config'])
        folder=OUT/c.experiment_id
        if folder.exists():raise FileExistsError('child refuses existing output directory')
        calibration=None if row['role']=='calibration' else read(JOB/'calibration.json')
        settings=settings_for(row,calibration)
        dump(JOB/'current-cell.json',dict(pid=os.getpid(),index=index,experiment=c.experiment_id))
        torch.set_num_threads(2)
        runner=BenchmarkRunner(c,folder)
        runner.challenge_arm,runner.challenge_thresholds=settings['arm'],settings['thresholds']
        dump(folder/'diagnostic-inputs.json',parent.EMPTY_BINDING)
        dump(folder/'repair-settings.json',settings)
        data=DATA/f'{c.dataset}-{c.seed}'
        try:
            x,y,ex,ey=[np.load(data/f'{n}.npy',allow_pickle=False) for n in
                ('features','labels','development_features','development_labels')]
            runner.run_arrays(x,y,lambda:create_model(c.dataset,x.shape[1:],c.num_classes,c.model),(ex,ey))
            if inputs()!=frozen:raise ValueError('inputs changed during child cell')
            events=[]
            with (folder/'events.jsonl').open(encoding='utf-8') as handle:
                for line in handle:
                    event=json.loads(line)
                    if event['event_type']=='model_integrity':events.append(event)
            if [e['round_index'] for e in events]!=list(range(1,c.rounds+1)):
                raise ValueError('missing integrity event')
            if not all(e['payload']['passed'] and e['payload']['expected_hash']==e['payload']['committed_hash'] for e in events):
                raise ValueError('invalid integrity event')
            dump(folder/'integrity-summary.json',dict(passed=True,checked_rounds=len(events),
                worker_pid=os.getpid(),cpu_arithmetic_reference=True,validated_commit_binding=True))
            dump(folder/'completion.json',sealed_files(folder))
            verify_cell(folder,c,settings)
        except IntegrityError as error:
            torch.save(error.evidence,folder/'integrity-failure.pt')
            dump(folder/'integrity-failure.json',dict(error=str(error),traceback=traceback.format_exc()))
            raise


def run():
    with exclusive_lock(JOB/'coordinator.lock'),threadpool_limits(limits=2):
        frozen = read(JOB/'frozen-inputs.json')
        if inputs() != frozen:
            raise ValueError('frozen inputs changed')
        with exclusive_lock(v14.JOB/'coordinator.lock'):
            pass
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required for long experiment')
        if read(v14.JOB/'status.json')['status'] not in ('completed','completed_with_gaps'):
            raise RuntimeError('V14 has not stopped')
        torch.set_num_threads(2)
        dump(JOB/'environment.json',dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,cpu_threads=2))
        state = dict(status='running',pid=os.getpid(),total=len(manifest()),completed=0,held=0,failed=0,
            rounds_per_cell=ROUNDS,started_utc=datetime.now(timezone.utc).isoformat(),publication_ready=False)
        dump(JOB/'status.json',state)
        records,calibration = [],None
        try:
            for cell_index,row in enumerate(frozen['manifest']):
                if inputs() != frozen:
                    raise ValueError('frozen inputs changed during experiment')
                if row['role'] != 'calibration' and calibration is None:
                    calibration = calibrate()
                if calibration is not None and read(JOB/'calibration.json') != calibration:
                    raise ValueError('calibration changed')
                c = ExperimentConfig(**row['config'])
                folder = OUT/c.experiment_id
                settings = settings_for(row,calibration)
                state.update(current=c.experiment_id,phase=row['role'],updated_utc=datetime.now(timezone.utc).isoformat())
                dump(JOB/'status.json',state)
                print(f"START {c.experiment_id} ({state['completed']} completed)",flush=True)
                if c.attack != 'none' and row['arm'] in ('current','aggregate','repair') and not gate(records,row['arm'])['passed']:
                    dump(folder/'held.json',gate(records,row['arm']))
                    state['held'] += 1
                    print(f'HOLD {c.experiment_id}: clean gate failed',flush=True)
                    analyze(records,state)
                    dump(JOB/'status.json',state)
                    continue
                try:
                    if (folder/'completion.json').exists():
                        summary = verify_cell(folder,c,settings)
                    else:
                        if folder.exists():
                            target = folder.with_name(folder.name+f'.interrupted-{time.time_ns()}')
                            if folder.resolve().parent != OUT.resolve() or target.resolve().parent != OUT.resolve():
                                raise ValueError('invalid result path')
                            folder.rename(target)
                        subprocess.run([sys.executable,'-u','-m','scripts.trust_repair_v16.study',
                            '--cell-index',str(cell_index)],cwd=ROOT,check=True,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                        if inputs()!=frozen:raise ValueError('frozen inputs changed inside cell')
                        summary=verify_cell(folder,c,settings)
                    if row['role'] != 'calibration':
                        enriched = enrich(summary,folder,row['arm'])
                        if row['arm'] == 'clip_only' and c.attack == 'none':
                            old = read(OUT/config(c.dataset,'clip_only',c.seed,calibration=True).experiment_id/'chain-trace.json')
                            new = read(folder/'chain-trace.json')
                            if [r['model_hash'] for r in old] != [r['model_hash'] for r in new]:
                                raise ValueError('passive calibration changed clipping trajectory')
                        records.append(enriched)
                    state['completed'] += 1
                    print(f"DONE {c.experiment_id}: accuracy={summary['final_utility']:.4f}",flush=True)
                except Exception:
                    state['failed'] += 1
                    error = traceback.format_exc()
                    dump(JOB/f'failure-{c.experiment_id}.json',dict(error=error))
                    print(error,flush=True)
                    raise
                analyze(records,state)
                dump(JOB/'status.json',state)
            state['status'] = 'completed_with_gaps' if state['held'] or state['failed'] else 'completed'
        except BaseException:
            state.update(status='failed',error=traceback.format_exc())
            raise
        finally:
            state['finished_utc'] = datetime.now(timezone.utc).isoformat()
            dump(JOB/'status.json',state)
            analyze(records,state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--cell-index',type=int)
    args = parser.parse_args()
    if args.cell_index is not None:
        execute_cell(args.cell_index)
    elif args.freeze:
        freeze()
        print(f'Frozen {len(manifest())} maximum 100-round cells')
    elif args.run:
        run()
    else:
        print(json.dumps(manifest(),indent=2))
