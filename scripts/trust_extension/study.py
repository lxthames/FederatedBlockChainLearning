"""Frozen follow-up study: FedVal, natural clients, adaptive attacks and recovery controls."""
import argparse
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from .femnist import digest, dump, prepare
from .research.config import ExperimentConfig
from .research.dataio import load_experiment_data
from .research.models import create_model
from .research.runner_v2 import BenchmarkRunner

ROOT = Path(__file__).resolve().parents[2]
JOB = ROOT/'artifacts/trust-extension-v1'
RESULTS = ROOT/'results/trust-extension-v1'
CONFIRM = (13,29,47,71,101)
DEVELOPMENT = (3,5,7)
METHODS = ('fedavg','fltrust','fedval','validation_only','full')
CONDITIONS = ('none','model_replacement','low_and_slow')


def code_hashes():
    paths = sorted(Path(__file__).parent.rglob('*.py'))
    return {str(p.relative_to(ROOT)):digest(p) for p in paths}


def cell(family, dataset, method, attack, seed, rounds=100, **changes):
    natural = str(ROOT/f'data/femnist-tff/seed-{seed}/natural-splits.json') if dataset=='femnist' else None
    c = ExperimentConfig(experiment_id='pending', dataset=dataset,num_clients=50,
        participation_rate=0.4,malicious_fraction=0.3,dirichlet_alpha=None if natural else 0.5,
        rounds=rounds,local_epochs=1,seed=seed,attack=attack,
        defense='two_layer' if method in {'validation_only','full'} else method,
        ablation='validation_only' if method=='validation_only' else 'full',
        device='cpu',natural_splits_path=natural,sampling_policy='bounded',
        num_classes=62 if natural else 10, evidence_tier='development' if family=='development' else 'confirmatory')
    c=replace(c,**changes)
    key=hashlib.sha256(json.dumps(c.to_dict(),sort_keys=True).encode()).hexdigest()[:10]
    c=replace(c,experiment_id=f'{family}-{dataset}-{method}-{attack}-s{seed}-{key}')
    return {'family':family,'method':method,'config':c.to_dict()}


def manifest():
    rows=[]
    for dataset,seed in itertools.product(('fashion_mnist','femnist'),DEVELOPMENT):
        for attack in CONDITIONS:
            rows.append(cell('development',dataset,'fedavg',attack,seed,rounds=20))
        for method in ('fedval','full'):
            rows.append(cell('development',dataset,method,'none',seed,rounds=20))
    for seed in DEVELOPMENT:
        rows.append(cell('development','fashion_mnist','fedavg','adaptive_model_replacement',seed,rounds=20))
    for dataset,method,attack,seed in itertools.product(('fashion_mnist','femnist'),METHODS,CONDITIONS,CONFIRM):
        rows.append(cell('matched_baselines',dataset,method,attack,seed))
    for method,seed in itertools.product(METHODS,CONFIRM):
        rows.append(cell('adaptive','fashion_mnist',method,'adaptive_model_replacement',seed))
    for behavior,fraction,attack,seed in itertools.product(('false_recovery',),(0.0,0.3,0.7),('none','low_and_slow'),CONFIRM):
        rows.append(cell('validator_stress','fashion_mnist','full',attack,seed,
            malicious_validator_fraction=fraction,validator_behavior=behavior))
    for fraction,seed in itertools.product((0.3,0.7),CONFIRM):
        rows.append(cell('validator_stress','fashion_mnist','full','none',seed,
            malicious_validator_fraction=fraction,validator_behavior='withhold'))
    for policy,attack,seed in itertools.product(('continue','scheduled_plain','scheduled_accountable'),('model_replacement','low_and_slow'),CONFIRM):
        rows.append(cell('recovery_controls','fashion_mnist','full',attack,seed,
            recovery_policy=policy,attack_start_round=6,intervention_round=20,intervention_checkpoint=5))
    for method,attack,seed in itertools.product(('full','multi_krum'),CONDITIONS,CONFIRM):
        rows.append(cell('uniform_participation','fashion_mnist',method,attack,seed,sampling_policy='uniform'))
    return rows


def load(c):
    if c.dataset=='femnist':
        folder=ROOT/f'data/femnist-tff/seed-{c.seed}'
        a=[np.load(folder/f'{name}.npy',allow_pickle=False) for name in
           ('features','labels','test_features','test_labels')]
        return a[0],a[1],(a[2],a[3])
    return load_experiment_data(c,ROOT/'data')


def execute(c,output):
    if output.exists() and (output/'summary.json').exists():
        result=json.loads((output/'summary.json').read_text())
        expected=hashlib.sha256(json.dumps(c.to_dict(),sort_keys=True).encode()).hexdigest()
        if result['config_hash']!=expected:
            raise ValueError('completed output configuration mismatch')
        return result
    if output.exists() and any(output.iterdir()):
        output.rename(output.with_name(output.name+f'.interrupted-{time.time_ns()}'))
    x,y,test=load(c)
    factory=lambda:create_model(c.dataset,x.shape[1:],62 if c.dataset=='femnist' else int(y.max())+1,c.model)
    result=BenchmarkRunner(c,output).run_arrays(x,y,factory,test)
    if c.dataset=='femnist':
        dump(output/'data-identity.json',{'dataset':'FEMNIST-TFF-62','provenance_sha256':digest(ROOT/'data/femnist-tff/provenance.json'),
            'natural_writer_splits':True,'actual_poisoning_fraction':len(result['malicious_client_ids'])/c.num_clients})
    return result


def freeze(rows):
    JOB.mkdir(parents=True,exist_ok=True)
    if not (ROOT/'data/femnist-tff/provenance.json').exists():
        prepare()
    data=json.loads((ROOT/'data/femnist-tff/provenance.json').read_text())
    for seed in data['seeds']:
        for name,value in seed['files'].items():
            if digest(ROOT/f"data/femnist-tff/seed-{seed['seed']}"/name)!=value:
                raise ValueError('FEMNIST generated data fingerprint mismatch')
    frozen={'code':code_hashes(),'manifest':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),
        'data':digest(ROOT/'data/femnist-tff/provenance.json')}
    path=JOB/'frozen-inputs.json'
    if path.exists() and json.loads(path.read_text())!=frozen:
        raise ValueError('follow-up study inputs changed; use a new version/output root')
    dump(JOB/'manifest.json',rows);dump(path,frozen)
    dump(JOB/'environment.json',{'python':sys.version,'executable':sys.executable,
        'platform':platform.platform(),'torch':torch.__version__,'numpy':np.__version__,
        'device':'cpu','torch_threads':torch.get_num_threads(),
        'timing_limitation':'concurrent background workloads; exploratory wall time, not isolated overhead'})
    with (JOB/'pip-freeze.txt').open('w',encoding='utf-8') as out:
        subprocess.run([sys.executable,'-m','pip','freeze'],stdout=out,check=True)
    return frozen


def gate(development, dataset, attack):
    controls={s['seed']:s for s in development if s['dataset']==dataset and s['defense']=='fedavg' and s['attack']=='none'}
    if len(controls)!=3:
        return {'passed':False,'reason':'incomplete clean development controls'}
    # Screen gross convergence failure; this is not an accuracy claim.
    floor = 2/62 if dataset == 'femnist' else 0.10
    if float(np.mean([s['final_utility'] for s in controls.values()])) <= floor:
        return {'passed':False,'reason':f'clean development utility did not exceed prespecified {floor:.6f} feasibility floor'}
    if attack=='none': return {'passed':True,'reason':'three clean controls completed'}
    attacks={s['seed']:s for s in development if s['dataset']==dataset and s['defense']=='fedavg' and s['attack']==attack}
    if set(attacks)!=set(DEVELOPMENT):return {'passed':False,'reason':'incomplete attacked development seeds'}
    targeted=attack in {'model_replacement','adaptive_model_replacement'}
    values=[((attacks[s]['mean_targeted_attack_success'] or 0)-(controls[s]['mean_clean_trigger_prediction_rate'] or 0))
        if targeted else controls[s]['final_utility']-attacks[s]['final_utility'] for s in DEVELOPMENT]
    threshold=0.05 if targeted else 0.005
    return {'passed':bool(np.mean(values)>threshold),'effects':values,'threshold':threshold,
            'criterion':'three matched development seeds; mean targeted-ASR lift or utility loss'}


METRICS=('final_utility','lower_decile_client_utility','worst_client_utility',
    'mean_targeted_attack_success','detection_false_positive_rate','detection_false_negative_rate',
    'detection_delay_rounds','recovery_count','wrongful_slashing_count','honest_validator_penalty_amount',
    'unsupported_rounds','monitoring_quorum_failures','runtime_seconds_total','transmitted_bytes','ledger_bytes')


def group_key(row):
    c=row['config']
    return (row['family'],c['dataset'],row['method'],c['attack'],c['recovery_policy'],
            c['validator_behavior'],c['malicious_validator_fraction'])


def analyze(rows):
    groups=defaultdict(list);completeness=[]
    for row in rows:
        c=row['config'];path=RESULTS/row['family']/c['experiment_id']/'summary.json'
        if row['family']=='development':continue
        if path.exists():groups[group_key(row)].append(json.loads(path.read_text()))
    expected=sorted({group_key(row) for row in rows if row['family']!='development'})
    analysis=[]
    for key in expected:
        runs=groups[key];seeds=sorted(s['seed'] for s in runs)
        complete=seeds==list(CONFIRM)
        completeness.append({'group':key,'seeds':seeds,'complete':complete})
        if not complete:continue
        record={'group':key,'seeds':seeds,'metrics':{}}
        for metric in METRICS:
            values=[r.get(metric) for r in runs]
            if all(isinstance(v,(int,float)) for v in values):
                values=np.asarray(values,dtype=float)
                rng=np.random.default_rng(202710)
                samples=rng.choice(values,(10000,5),replace=True).mean(1)
                record['metrics'][metric]={'mean':float(values.mean()),'bootstrap_ci95':np.quantile(samples,[.025,.975]).tolist(),'seed_values':values.tolist()}
        analysis.append(record)
    paired=[]
    for row in analysis:
        key=tuple(row['group'])
        if key[2]!='full':continue
        for comparison in analysis:
            other=tuple(comparison['group'])
            if key[:2]+key[3:] != other[:2]+other[3:] or other[2]=='full':continue
            left={s['seed']:s for s in groups[key]};right={s['seed']:s for s in groups[other]}
            values=np.array([left[s]['final_utility']-right[s]['final_utility'] for s in CONFIRM])
            rng=np.random.default_rng(202711)
            ci=np.quantile(rng.choice(values,(10000,5),replace=True).mean(1),[.025,.975]).tolist()
            paired.append({'full':key,'comparison':other,'seeds':list(CONFIRM),
                           'paired_utility_differences':values.tolist(),'mean':float(values.mean()),'bootstrap_ci95':ci})
    recovery_pairs=[]
    for key,runs in groups.items():
        if key[0]!='recovery_controls' or key[4]=='continue' or sorted(s['seed'] for s in runs)!=list(CONFIRM):continue
        control_key=key[:4]+('continue',)+key[5:]
        controls={s['seed']:s for s in groups.get(control_key,[])}
        if set(controls)!=set(CONFIRM):continue
        entries=[]
        for result in sorted(runs,key=lambda s:s['seed']):
            control=controls[result['seed']];round_index=result['configuration']['intervention_round']-1
            entries.append({'seed':result['seed'],
                'before_intervention_histories_match':result['accuracy_history'][:round_index]==control['accuracy_history'][:round_index],
                'final_utility_difference':result['final_utility']-control['final_utility'],
                'mean_post_intervention_utility_difference':float(np.mean(np.array(result['accuracy_history'][round_index:])-np.array(control['accuracy_history'][round_index:]))),
                'wrongful_slashing_count':result['wrongful_slashing_count']})
        if not all(e['before_intervention_histories_match'] for e in entries):
            raise AssertionError('recovery-only comparison diverged before scheduled intervention')
        recovery_pairs.append({'intervention':key,'control':control_key,'seed_pairs':entries})
    dump(JOB/'five-seed-analysis.json',{'complete_groups':analysis,'paired_comparisons':paired,
        'recovery_only_comparisons':recovery_pairs,
        'uncertainty_note':'Five seeds give limited precision; bootstrap intervals are exploratory; no significance claim from CI alone'})
    dump(JOB/'completeness.json',completeness)


def runtime_forecast(rows):
    path=JOB/'runtime-preflight.json'
    if not path.exists():return
    benchmarks=json.loads(path.read_text())
    times={(r['dataset'],r['method']):r['round_seconds'] for r in benchmarks}
    nominal=0.0
    for row in rows:
        c=row['config'];method='full' if row['method']=='validation_only' else row['method']
        seconds=times.get((c['dataset'],method),times[(c['dataset'],'fedavg')])
        if c['attack']=='adaptive_model_replacement':seconds*=2
        nominal+=c['rounds']*seconds
    dump(JOB/'runtime-forecast.json',{'nominal_hours':nominal/3600,
        'planning_range_hours':[nominal/3600,2*nominal/3600],
        'assumptions':'Linear extrapolation from concurrent one-round CPU preflight; adaptive rounds factor two; Multi-Krum approximated by FedAvg; excludes deployment replay/setup; not a completion guarantee',
        'preflight_data_note':'Preflight predates FEMNIST duplicate removal; preliminary runtime only, never scientific evidence'})


def replay_one(c,output):
    from .research.evm import EVMAdapter
    trace=json.loads((output/'chain-trace.json').read_text())
    replay=output/'evm-replay'
    if (replay/'agreement.json').exists():return
    if replay.exists() and any(replay.iterdir()):
        replay.rename(replay.with_name(replay.name+f'.interrupted-{time.time_ns()}'))
    adapter=EVMAdapter(replace(c,experiment_id=c.experiment_id+'-replay'),replay)
    started=time.perf_counter();adapter.genesis(trace[0])
    for record in trace[1:]:adapter.round(record)
    summary=json.loads((output/'summary.json').read_text())
    for client,stake in summary['final_stakes'].items():
        if adapter.contract.functions.stake(adapter.account(int(client))).call()!=adapter.units(stake):
            raise AssertionError('final replay stake disagrees with simulator')
    dump(replay/'agreement.json',{'status':'passed','trace_sha256':digest(output/'chain-trace.json'),
        'rounds':len(trace)-1,'elapsed_replay_seconds':time.perf_counter()-started,
        'active_version':summary['active_version'],'stake_agreement':True,
        'claim_limit':'local execution on identical decisions, not proof that a ledger improves learning quality'})


def run():
    torch.set_num_threads(2)
    rows=manifest();frozen=freeze(rows);runtime_forecast(rows)
    state={'status':'running','pid':os.getpid(),'completed':0,'failed':0,'systems_failed':0,'held':0,
        'started_utc':datetime.now(timezone.utc).isoformat(),'publication_ready':False,
        'development_total':sum(r['family']=='development' for r in rows),
        'evaluation_total':sum(r['family']!='development' for r in rows),'device':'cpu'}
    def status():
        state['updated_utc']=datetime.now(timezone.utc).isoformat();dump(JOB/'status.json',state)
    development=[];held=[];gates={};replayed=False
    try:
        for row in rows:
            c=ExperimentConfig(**row['config']);state.update(phase=row['family'],current_experiment=c.experiment_id);status()
            if code_hashes()!=frozen['code']:raise ValueError('frozen extension code changed during execution')
            if row['family']!='development':
                key=f'{c.dataset}:{c.attack}'
                if key not in gates:gates[key]=gate(development,c.dataset,c.attack)
                dump(JOB/'attack-validation.json',gates)
                if not gates[key]['passed']:
                    held.append({'experiment_id':c.experiment_id,'family':row['family'],'reason':gates[key]})
                    state['held']+=1;dump(JOB/'held-experiments.json',held);status();continue
            output=RESULTS/row['family']/c.experiment_id
            try:
                started=time.perf_counter();summary=execute(c,output)
                state['last_run_seconds']=time.perf_counter()-started
                state['completed']+=1
                if row['family']=='development':development.append(summary)
                if not replayed and row['family']=='development' and row['method']=='full':
                    try:replay_one(c,output);replayed=True
                    except Exception:
                        state['systems_failed']+=1
                        with (JOB/'systems-failures.jsonl').open('a',encoding='utf-8') as out:
                            out.write(json.dumps({'experiment_id':c.experiment_id,'error':traceback.format_exc()})+'\n')
                if row['family']!='development':analyze(rows)
            except Exception:
                state['failed']+=1
                with (JOB/'failures.jsonl').open('a',encoding='utf-8') as out:
                    out.write(json.dumps({'config':c.to_dict(),'family':row['family'],'error':traceback.format_exc()})+'\n')
            status()
        analyze(rows)
        state['status']='completed_with_gaps' if state['failed'] or state['systems_failed'] or held else 'completed'
    except BaseException:
        state['status']='failed';state['error']=traceback.format_exc();raise
    finally:
        state['finished_utc']=datetime.now(timezone.utc).isoformat();status()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('run','manifest'))
    args=parser.parse_args()
    if args.command=='run':run()
    else:
        rows=manifest();dump(JOB/'manifest-preview.json',rows)
        print(json.dumps({'development':sum(r['family']=='development' for r in rows),
                          'evaluation':sum(r['family']!='development' for r in rows)}))
