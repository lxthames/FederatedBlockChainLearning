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
from dataclasses import replace
import numpy as np
from .cli import _execute_config
from .trust_matrix import Trust2027Matrix, cell, DEVELOPMENT_SEEDS, ATTACKS

ROOT=Path(__file__).resolve().parents[2]
JOB=ROOT/'artifacts/trust-2027'
RESULTS=ROOT/'results/trust-2027'


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True),encoding='utf-8')
    temporary.replace(path)


def prepare():
    matrix=Trust2027Matrix()
    for name in ('development','calibration','core_confirmation','scale_sweep','heterogeneity_sweep',
                 'attacker_fraction_sweep','identity_committee_sweep','ablations','benign_drift','sift_case_study'):
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
    dump(JOB/'source-fingerprints.json',source)


def execute(config,stage):
    output=RESULTS/stage/config.experiment_id
    if (output/'summary.json').exists():
        summary=json.loads((output/'summary.json').read_text())
        expected=hashlib.sha256(json.dumps(config.to_dict(),sort_keys=True).encode()).hexdigest()
        if summary.get('config_hash')!=expected: raise ValueError('resume configuration mismatch')
        source_hash=hashlib.sha256(b''.join(p.read_bytes() for p in sorted((ROOT/'scripts/research').glob('*.py')))).hexdigest()
        if summary.get('source_hash')!=source_hash:
            raise ValueError('completed run was produced by different research code; use a new study directory')
        return summary
    if output.exists() and any(output.iterdir()):
        output.rename(output.with_name(output.name+f'.interrupted-{time.time_ns()}'))
    return _execute_config(config,output,ROOT/'data')


def run():
    prepare()
    state={'status':'running','pid':os.getpid(),'completed':0,'failed':0,
           'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'stage':'development','current_experiment':None,'publication_ready':False}
    def status():
        state['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        dump(JOB/'status.json',state)
    def attempt(config,stage):
        state['stage']=stage; state['current_experiment']=config.experiment_id; status()
        try:
            result=execute(config,stage)
            state['completed']+=1
            return result
        except Exception:
            state['failed']+=1
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
        stages=('core_confirmation','scale_sweep','heterogeneity_sweep','attacker_fraction_sweep',
                'identity_committee_sweep','ablations','benign_drift')
        for stage in stages:
            path=JOB/'manifests'/f'{stage}.json'
            if hashlib.sha256(path.read_bytes()).hexdigest()!=frozen[path.name]:
                raise ValueError('frozen manifest changed during execution')
            for item in json.loads(path.read_text()):
                from .config import ExperimentConfig
                config=ExperimentConfig(**item)
                if not gate[f'{config.dataset}:{config.attack}']['passed']:
                    held.append({'stage':stage,'experiment_id':config.experiment_id,'reason':'attack not validated on development seeds'})
                    continue
                attempt(config,stage)
            dump(JOB/'held-experiments.json',held)
            subprocess.run([sys.executable,'-m','scripts.research.paper_artifacts'],cwd=ROOT,check=True)
        state['status']='completed_with_gaps' if state['failed'] or held else 'completed'
    except BaseException:
        state['status']='failed'; state['error']=traceback.format_exc()
        raise
    finally:
        state['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        status()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run'))
    args=parser.parse_args()
    prepare() if args.command=='prepare' else run()

if __name__=='__main__': main()
