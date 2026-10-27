"""Prepare, freeze and execute the TRUST study with an observable background status."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import shutil
from dataclasses import replace
import numpy as np
from .cli import _execute_config
from .trust_matrix import Trust2027Matrix, cell, DEVELOPMENT_SEEDS, ATTACKS

ROOT=Path(__file__).resolve().parents[2]
JOB=ROOT/'artifacts/trust-2027'
RESULTS=ROOT/'results/trust-2027'
STAGES=('development','calibration','core_confirmation','scale_sweep','heterogeneity_sweep',
        'attacker_fraction_sweep','identity_committee_sweep','ablations','benign_drift','sift_case_study')


def source_hash():
    return hashlib.sha256(b''.join(p.read_bytes() for p in sorted((ROOT/'scripts/research').glob('*.py')))).hexdigest()


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True),encoding='utf-8')
    temporary.replace(path)


def prepare():
    matrix=Trust2027Matrix()
    for name in STAGES:
        path=JOB/'manifests'/f'{name}.json'
        expected=[c.to_dict() for c in getattr(matrix,name)()]
        if path.exists() and json.loads(path.read_text())!=expected:
            if (JOB/'frozen-manifests.json').exists():
                raise ValueError('cannot replace a frozen study manifest')
            dump(path,expected)
        elif not path.exists(): dump(path,expected)
    dump(JOB/'environment.json',{'python':sys.version,'executable':sys.executable})
    with (JOB/'pip-freeze.txt').open('w',encoding='utf-8') as out:
        subprocess.run([sys.executable,'-m','pip','freeze'],stdout=out,check=True)
    source={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT/'scripts/research').glob('*.py')}
    frozen_path=JOB/'frozen-run.json'
    frozen=dict(source_hash=source_hash(),source=source,
        manifests={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (JOB/'manifests').glob('*.json')})
    if frozen_path.exists() and json.loads(frozen_path.read_text())!=frozen:
        raise ValueError('frozen source or manifests changed; use a new study name')
    if not frozen_path.exists():
        for p in (ROOT/'scripts/research').glob('*.py'):
            destination=JOB/'source-snapshot'/p.name
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,destination)
        dump(frozen_path,frozen)
    dump(JOB/'source-fingerprints.json',source)


def execute(config,stage):
    output=RESULTS/stage/config.experiment_id
    if (output/'summary.json').exists():
        summary=json.loads((output/'summary.json').read_text())
        expected=hashlib.sha256(json.dumps(config.to_dict(),sort_keys=True).encode()).hexdigest()
        if summary.get('config_hash')!=expected: raise ValueError('resume configuration mismatch')
        if summary.get('source_hash') != source_hash():
            raise ValueError('completed run was produced by different research code; use a new study directory')
        if summary.get('status')!='completed' or summary.get('rounds_completed')!=config.rounds:
            raise ValueError('existing summary is incomplete')
        return summary
    if output.exists() and any(output.iterdir()):
        output.rename(output.with_name(output.name+f'.interrupted-{time.time_ns()}'))
    return _execute_config(config,output,ROOT/'data')


def run():
    import torch
    torch.set_num_threads(2)
    prepare()
    frozen_run=json.loads((JOB/'frozen-run.json').read_text())
    state={'status':'running','pid':os.getpid(),'completed':0,'failed':0,
           'held':0,'analysis_errors':[],
           'total':sum(len(getattr(Trust2027Matrix(),name)()) for name in STAGES),
           'source_hash':frozen_run['source_hash'],'results_root':str(RESULTS),
           'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'stage':'development','current_experiment':None,'publication_ready':False}
    def status():
        state['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        dump(JOB/'status.json',state)
    def attempt(config,stage):
        if source_hash()!=frozen_run['source_hash']:
            raise ValueError('source changed while running; stopped to preserve comparable results')
        state['stage']=stage; state['current_experiment']=config.experiment_id; status()
        print(f'START {stage} {config.experiment_id}',flush=True)
        try:
            result=execute(config,stage)
            state['completed']+=1
            print(f'DONE {state["completed"]}/{state["total"]} {config.experiment_id}',flush=True)
            return result
        except Exception:
            state['failed']+=1
            print(f'FAILED {config.experiment_id}: {traceback.format_exc()}',flush=True)
            path=JOB/'failures.jsonl'
            with path.open('a',encoding='utf-8') as out:
                out.write(json.dumps({'stage':stage,'config':config.to_dict(),'error':traceback.format_exc()})+'\n')
            return None
        finally: status()
    status()
    try:
        development=[]
        for config in Trust2027Matrix().development():
            summary=attempt(config,'development')
            if summary: development.append(summary)
        # The default thresholds are prespecified; this grid measures sensitivity
        # on development seeds without selecting against confirmatory outcomes.
        for config in Trust2027Matrix().calibration():
            attempt(config,'calibration')
        # Freeze actual manifests before looking at confirmatory outcomes.
        frozen={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (JOB/'manifests').glob('*.json')}
        dump(JOB/'frozen-manifests.json',frozen)
        gate={}
        for dataset in ('fashion_mnist','cifar10'):
            avg=[s for s in development if s['dataset']==dataset and s['defense']=='fedavg']
            clean={s['seed']:s for s in avg if s['attack']=='none'}
            for attack in ATTACKS:
                rows=[s for s in avg if s['attack']==attack and s['seed'] in clean]
                differences=[]
                for row in rows:
                    baseline=clean[row['seed']]
                    if attack=='model_replacement':
                        differences.append((row.get('mean_targeted_attack_success') or 0.0)
                                           -(baseline.get('mean_clean_trigger_prediction_rate') or 0.0))
                    else:
                        differences.append(baseline['final_utility']-row['final_utility'])
                validated=attack=='none' or (len(rows)==3 and float(np.mean(differences))>0.005)
                gate[f'{dataset}:{attack}']={'passed':validated,'development_seed_count':len(rows),
                    'effect_values':differences,'criterion':'clean control; otherwise 3 matched development seeds with mean utility loss >0.005, or targeted ASR lift >0.05 over clean-trigger control'}
                if attack=='model_replacement': gate[f'{dataset}:{attack}']['passed']=len(rows)==3 and float(np.mean(differences))>0.05
        dump(JOB/'attack-validation.json',gate)
        held=[]
        stages=STAGES[2:]
        for stage in stages:
            path=JOB/'manifests'/f'{stage}.json'
            if hashlib.sha256(path.read_bytes()).hexdigest()!=frozen[path.name]:
                raise ValueError('frozen manifest changed during execution')
            for item in json.loads(path.read_text()):
                from .config import ExperimentConfig
                config=ExperimentConfig(**item)
                reason=None
                if config.dataset=='sift_routing':
                    audit_path=ROOT/'data/sift-provenance.json'
                    if not audit_path.exists() or not json.loads(audit_path.read_text()).get('query_target_correspondence_verified'):
                        reason='SIFT query/target provenance is missing or unverified'
                elif not gate[f'{config.dataset}:{config.attack}']['passed']:
                    reason='attack not validated on development seeds'
                if reason:
                    held.append({'stage':stage,'experiment_id':config.experiment_id,'reason':reason})
                    state['held']=len(held)
                    status()
                    continue
                attempt(config,stage)
            dump(JOB/'held-experiments.json',held)
            result=subprocess.run([sys.executable,'-m','scripts.research.paper_artifacts',
                '--results-root',str(RESULTS),'--paper-root',str(JOB/'paper')],cwd=ROOT)
            if result.returncode:
                state['analysis_errors'].append(dict(stage=stage,exit_code=result.returncode))
                status()
        state['status']='completed_with_gaps' if state['failed'] or held or state['analysis_errors'] else 'completed'
    except BaseException:
        state['status']='failed'; state['error']=traceback.format_exc()
        raise
    finally:
        state['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        status()


def main():
    global JOB,RESULTS
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run'))
    parser.add_argument('--study-name',default='trust-2027')
    args=parser.parse_args()
    if not args.study_name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.study_name):
        parser.error('study-name must contain only letters, digits, hyphens and underscores')
    JOB=ROOT/'artifacts'/args.study_name
    RESULTS=ROOT/'results'/args.study_name
    JOB.mkdir(parents=True,exist_ok=True)
    with (JOB/'workflow.lock').open('a+b') as lock:
        lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepare() if args.command=='prepare' else run()

if __name__=='__main__': main()
