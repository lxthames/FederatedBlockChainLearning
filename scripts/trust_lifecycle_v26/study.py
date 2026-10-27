"""Frozen 400-trajectory lifecycle study with source/data seals and GPU queue."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ.setdefault(name,'2')
import argparse
from dataclasses import replace
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from scripts.trust_lifecycle_v24.runner import BenchmarkRunner as BaseRunner
from scripts.trust_alignment_v17.study import config as old_config
from scripts.trust_repair_v16_fixed.research.models import create_model
from scripts.trust_repair_v16_fixed.integrity import snapshot,state_hash
from scripts.trust_repair_v16_fixed.runtime import exclusive_lock
from scripts.trust_alignment_v17.study import dump,digest,read
from scripts.trust_lifecycle_v26.baselines import RepairController
from .data import DATA,SEEDS,prepare,load

ROOT=Path(__file__).resolve().parents[2]
JOB=ROOT/'artifacts/lifecycle-v26'
OUT=ROOT/'results/lifecycle-v26'
ARMS=('quorum','periodic','nad','fltrust')
SETTINGS=(('clean','patch'),('attack_stop','patch'),('continuous','patch'),
          ('attack_stop','checkerboard'),('continuous','checkerboard'))


def manifest():
    return [dict(dataset=d,seed=s,condition=c,trigger=t,arm=a,
        id=f'v26-{d}-{a}-{c}-{t}-s{s}')
        for c,t in SETTINGS for d in ('femnist','fashion_mnist') for s in SEEDS for a in ARMS]


def config(row):
    # Inherit numerical hyperparameters, never old data bindings or seed guards.
    c=old_config(row['dataset'],'current',11,
        'none' if row['condition']=='clean' else 'model_replacement')
    # FLTrust is the client/update baseline; repair arms retain the V25 two-layer controller.
    if row['arm']=='fltrust': c=replace(c,defense='fltrust')
    return replace(c,experiment_id=row['id'],seed=row['seed'],rounds=100,
        natural_splits_path=str(DATA/f'femnist-{row["seed"]}'/'natural-splits.json') if row['dataset']=='femnist' else None)


class Runner(BaseRunner):
    trigger='patch'
    def _apply_trigger(self,features):
        if self.trigger=='patch': return super()._apply_trigger(features)
        if self.trigger!='checkerboard': raise ValueError('unknown trigger')
        result=features.copy()
        checker=(np.indices((6,6)).sum(0)%2).astype(np.float32)
        result[:,:,11:17,11:17]=checker
        return result


class Progress:
    def __init__(self,logger,folder): self.logger=logger;self.folder=folder
    def log(self,event_type,round_index,payload):
        self.logger.log(event_type,round_index,payload)
        if event_type=='round_completed':
            dump(self.folder/'progress.json',dict(round=round_index,total=100,updated_utc=datetime.now(timezone.utc).isoformat()))
            if round_index%10==0: print(f'ROUND {round_index}/100',flush=True)


def source_paths():
    # Freeze transitive local code too; exclude the independent mutable research workflow.
    return sorted(p for folder in (ROOT/'scripts').glob('trust_*') if folder.is_dir() for p in folder.rglob('*.py'))


def freeze():
    provenance=prepare()
    value=dict(manifest=[dict(r,config=config(r).to_dict()) for r in manifest()],
        source={str(p.relative_to(ROOT)):digest(p) for p in source_paths()},
        data=provenance['files'],data_provenance_sha256=digest(DATA/'provenance.json'),
        protocol_sha256=digest(ROOT/'docs/research/LIFECYCLE_V25.md'),
        launcher_sha256=digest(ROOT/'scripts/run_lifecycle_v25.ps1'),
        tests_sha256=digest(ROOT/'tests/test_lifecycle_v25.py'),
        publication_ready=False)
    path=JOB/'frozen-inputs.json'
    if path.exists() and read(path)!=value: raise ValueError('V25 already frozen with different inputs')
    if not path.exists():
        for p in source_paths():
            target=JOB/'source-snapshot'/p.relative_to(ROOT)
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
        dump(path,value)
    dump(JOB/'manifest.json',value['manifest'])
    return value


def check_frozen(check_data=False):
    frozen=read(JOB/'frozen-inputs.json')
    actual={str(p.relative_to(ROOT)):digest(p) for p in source_paths()}
    if actual!=frozen['source']: raise ValueError('frozen lifecycle source changed')
    if digest(ROOT/'docs/research/LIFECYCLE_V25.md')!=frozen['protocol_sha256']: raise ValueError('protocol changed')
    if digest(DATA/'provenance.json')!=frozen['data_provenance_sha256']: raise ValueError('data provenance changed')
    if check_data:
        for name,h in frozen['data'].items():
            if digest(DATA/name)!=h: raise ValueError(f'data changed: {name}')
    return frozen


def seal(folder):
    return {str(p.relative_to(folder)):digest(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='completion.json'}


def execute(index):
    check_frozen(check_data=True)
    torch.set_num_threads(2)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    row=manifest()[index];c=config(row);folder=OUT/row['id']
    if folder.exists(): raise FileExistsError(folder)
    x,y,tx,ty=load(row['dataset'],row['seed'])
    runner=Runner(c,folder);runner.trigger=row['trigger']
    runner.attack_stop_round=50 if row['condition']=='attack_stop' else 100
    runner.challenge_arm='clip_only' 'clip_only' if row['arm']=='clipping' else 'current'
    runner.logger=Progress(runner.logger,folder)
    if row['arm'] in ('periodic','nad'):
        runner.online_controller=RepairController(row['arm'],False)
    # Evaluation callback runs only after admission. Outputs never enter repair.
    offline=[]
    def observer(outcome,rnd):
        records={}
        mask=ty!=c.backdoor_target_label
        for stage,model in (('source',outcome.source),('candidate',outcome.candidate),('committed',outcome.committed)):
            records[stage]=dict(model_hash=state_hash(snapshot(model)),accuracy=runner._score(model,tx,ty),
                trigger_rate=runner._score(model,runner._apply_trigger(tx[mask]),np.full(int(mask.sum()),c.backdoor_target_label,dtype=np.int64)))
        offline.append(dict(round=rnd,models=records,admitted=outcome.payload['decision']['passed']))
    runner.offline_observer=observer
    summary=runner.run_arrays(x,y,lambda:create_model(c.dataset,x.shape[1:],c.num_classes,c.model),(tx,ty))
    events=[json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
    integrity=[e for e in events if e['event_type']=='model_integrity']
    repairs=[e['payload'] for e in events if e['event_type']=='online_repair']
    if len(integrity)!=100 or not all(e['payload']['passed'] for e in integrity): raise ValueError('incomplete model integrity audit')
    expected=[25,50,75,100] if runner.online_controller else []
    if [r['round'] for r in repairs]!=expected: raise ValueError('repair schedule mismatch')
    history=summary['targeted_attack_success_history']
    if row['condition']!='clean' and len(history)!=100: raise ValueError('incomplete attack trajectory')
    record=dict(row,final_accuracy=summary['final_accuracy'],
        mean_post_onset_asr=float(np.mean(history[10:])) if history else None,
        mean_after_stop_asr=float(np.mean(history[50:])) if row['condition']=='attack_stop' else None,
        endpoint_asr=history[-1] if history else None,
        repair_admissions=sum(int(r['decision']['passed']) for r in repairs),
        repair_attempts=len(repairs),repair_seconds=sum(r['seconds'] for r in repairs),
        repair_work=[r['work'] for r in repairs],transmitted_bytes=summary['transmitted_bytes'],
        lower_decile_client_utility=summary['lower_decile_client_utility'],
        freeze_sha256=digest(JOB/'frozen-inputs.json'),integrity_passed=True)
    dump(folder/'lifecycle-summary.json',record)
    dump(folder/'offline-evaluation.json',offline)
    torch.save(snapshot(runner.final_model),folder/'final.pt')
    check_frozen()
    dump(folder/'completion.json',seal(folder))


def report():
    from scripts.trust_repair_v16_fixed.research.analysis import bootstrap_mean_ci
    records=[]
    for row in manifest():
        folder=OUT/row['id']
        if (folder/'completion.json').exists(): records.append(read(folder/'lifecycle-summary.json'))
    groups={}
    for row in records:
        key=(row['dataset'],row['condition'],row['trigger'],row['arm'])
        groups.setdefault(key,[]).append(row)
    aggregates=[]
    for key,rows in groups.items():
        if {r['seed'] for r in rows}!=set(SEEDS): continue
        entry=dict(zip(('dataset','condition','trigger','arm'),key),seed_count=5)
        for metric in ('final_accuracy','mean_post_onset_asr','mean_after_stop_asr','endpoint_asr','repair_admissions'):
            values=[r[metric] for r in rows]
            if all(v is not None for v in values):
                ci=bootstrap_mean_ci(values,seed=2501)
                entry[metric]=dict(mean=ci.mean,lower=ci.lower,upper=ci.upper)
        aggregates.append(entry)
    paired=[]
    for key,rows in groups.items():
        d,c,t,arm=key
        if arm=='quorum': continue
        baseline={r['seed']:r for r in groups.get((d,c,t,'quorum'),[])}
        treatment={r['seed']:r for r in rows}
        if set(baseline)!=set(SEEDS) or set(treatment)!=set(SEEDS): continue
        comparison=dict(dataset=d,condition=c,trigger=t,arm=arm,baseline='quorum',seed_count=5)
        for metric in ('final_accuracy','mean_post_onset_asr','mean_after_stop_asr','endpoint_asr'):
            if all(treatment[s][metric] is not None and baseline[s][metric] is not None for s in SEEDS):
                values=[treatment[s][metric]-baseline[s][metric] for s in SEEDS]
                ci=bootstrap_mean_ci(values,seed=2501)
                comparison[metric+'_difference']=dict(mean=ci.mean,lower=ci.lower,upper=ci.upper)
        paired.append(comparison)
    dump(JOB/'analysis.json',dict(completed=len(records),total=len(manifest()),records=records,
        complete_seed_groups=aggregates,paired_comparisons=paired))


def run():
    with exclusive_lock(JOB/'coordinator.lock'):
        check_frozen(check_data=True)
        state=dict(status='waiting_for_gpu',pid=os.getpid(),completed=0,failed=0,total=len(manifest()),
            started_utc=datetime.now(timezone.utc).isoformat(),publication_ready=False)
        def update():
            state['updated_utc']=datetime.now(timezone.utc).isoformat();dump(JOB/'status.json',state)
        try:
            update()
            # Queue behind the currently authorized GPU job, including its full lineage.
            while True:
                processes=subprocess.run(['powershell.exe','-NoProfile','-Command',
                    "@(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\\.exe$' -and $_.CommandLine -match 'scripts\\.research\\.workflow.*run' }).Count"],
                    capture_output=True,text=True,check=True)
                if int(processes.stdout.strip())==0: break
                state['waiting_for']='active TRUST workflow';update();time.sleep(30)
            state.pop('waiting_for',None);state['status']='running';update()
            for index,row in enumerate(manifest()):
                check_frozen();folder=OUT/row['id']
                state['current']=row['id'];update()
                if (folder/'completion.json').exists():
                    if read(folder/'completion.json')!=seal(folder): raise ValueError('completed output seal mismatch')
                else:
                    if folder.exists(): folder.rename(folder.with_name(folder.name+f'.interrupted-{time.time_ns()}'))
                    print(f'START {index+1}/{len(manifest())} {row["id"]}',flush=True)
                    result=subprocess.run([sys.executable,'-u','-m','scripts.trust_lifecycle_v26.study','--cell',str(index)],cwd=ROOT)
                    if result.returncode:
                        state['failed']+=1;update()
                        with (JOB/'failures.jsonl').open('a') as f: f.write(json.dumps(dict(row,index=index,exit_code=result.returncode))+'\n')
                        continue
                    if read(folder/'completion.json')!=seal(folder): raise ValueError('worker seal mismatch')
                state['completed']+=1;report();update()
            state['status']='completed_with_gaps' if state['failed'] else 'complete'
            state['finished_utc']=datetime.now(timezone.utc).isoformat();update()
        except BaseException:
            state.update(status='failed',error=traceback.format_exc());update();raise


def main():
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze',action='store_true');group.add_argument('--run',action='store_true')
    group.add_argument('--cell',type=int);group.add_argument('--report',action='store_true')
    args=parser.parse_args()
    if args.freeze: freeze();print('Frozen V26 baseline matrix')
    elif args.run: run()
    elif args.report: report()
    else: execute(args.cell)

if __name__=='__main__': main()

