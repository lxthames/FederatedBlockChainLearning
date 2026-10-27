"""GPU execution of the frozen TRUST extension design, with separate artifacts.

Reuses the original protocol implementation without modifying its source or
combining CPU and GPU seed groups. All development gates are rerun on CUDA.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback
import torch
from scripts.trust_extension import study
from scripts.trust_extension.femnist import digest, dump
from scripts.trust_extension.research.config import ExperimentConfig

ROOT = Path(__file__).resolve().parents[2]
JOB = ROOT/'artifacts/trust-extension-v1-gpu'
RESULTS = ROOT/'results/trust-extension-v1-gpu'


def gpu_manifest():
    rows=study.manifest()
    for row in rows:
        row['config']['device']='cuda'
        row['config']['experiment_id']+='-cuda'
    return rows


def forecast(rows):
    path=ROOT/'artifacts/trust-extension-gpu-preflight.json'
    if not path.exists():return
    timings={row['method']:row['round_seconds'] for row in json.loads(path.read_text())}
    seconds=0.0
    for row in rows:
        config=row['config']; method=row['method']
        duration=timings['full' if method in {'full','validation_only'} else 'fedavg']
        # No exact GPU FedVal/FEMNIST timing exists: conservative scheduling factors.
        if method=='fedval':duration*=2
        if config['attack']=='adaptive_model_replacement':duration*=2
        seconds+=duration*config['rounds']
    dump(JOB/'runtime-forecast.json',{'nominal_hours':seconds/3600,
        'planning_range_hours':[seconds/3600,seconds/1800],
        'assumptions':'Concurrent one-round Fashion-MNIST CUDA preflight; FedVal and adaptive factor two; FEMNIST and Multi-Krum approximated; excludes replay/setup. Planning estimate, not completion guarantee.'})


def run():
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; refusing CPU fallback')
    torch.set_num_threads(2)
    study.JOB,study.RESULTS=JOB,RESULTS
    rows=gpu_manifest()
    original_hashes=study.code_hashes
    # The inherited freeze must include this execution adapter too.
    def combined_hashes():
        values=original_hashes()
        values[str(Path(__file__).relative_to(ROOT))]=digest(__file__)
        return values
    study.code_hashes=combined_hashes
    frozen=study.freeze(rows)
    environment=json.loads((JOB/'environment.json').read_text())
    environment.update(device='cuda',gpu=torch.cuda.get_device_name(0),
                       cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'])
    dump(JOB/'environment.json',environment)
    forecast(rows)
    state={'status':'running','pid':os.getpid(),'device':'cuda','gpu':torch.cuda.get_device_name(0),
        'completed':0,'failed':0,'systems_failed':0,'held':0,'publication_ready':False,
        'development_total':sum(r['family']=='development' for r in rows),
        'evaluation_total':sum(r['family']!='development' for r in rows),
        'started_utc':datetime.now(timezone.utc).isoformat(),
        'cpu_results_preserved_at':'results/trust-extension-v1',
        'cpu_results_reused':False}
    def status():
        state['updated_utc']=datetime.now(timezone.utc).isoformat();dump(JOB/'status.json',state)
    development=[];held=[];gates={};replayed=False
    try:
        for row in rows:
            c=ExperimentConfig(**row['config'])
            state.update(phase=row['family'],current_experiment=c.experiment_id);status()
            if combined_hashes()!=frozen['code']:raise ValueError('frozen GPU study source changed')
            if row['family']!='development':
                key=f'{c.dataset}:{c.attack}'
                if key not in gates:gates[key]=study.gate(development,c.dataset,c.attack)
                dump(JOB/'attack-validation.json',gates)
                if not gates[key]['passed']:
                    held.append({'experiment_id':c.experiment_id,'family':row['family'],'reason':gates[key]})
                    state['held']+=1;dump(JOB/'held-experiments.json',held);status();continue
            output=RESULTS/row['family']/c.experiment_id
            try:
                started=time.perf_counter();summary=study.execute(c,output)
                if summary['device']!='cuda':raise AssertionError('run did not execute on CUDA')
                state['last_run_seconds']=time.perf_counter()-started;state['completed']+=1
                if row['family']=='development':development.append(summary)
                if not replayed and row['family']=='development' and row['method']=='full':
                    try:study.replay_one(c,output);replayed=True
                    except Exception:
                        state['systems_failed']+=1
                        with (JOB/'systems-failures.jsonl').open('a',encoding='utf-8') as out:
                            out.write(json.dumps({'experiment_id':c.experiment_id,'error':traceback.format_exc()})+'\n')
                if row['family']!='development':study.analyze(rows)
            except Exception:
                state['failed']+=1
                with (JOB/'failures.jsonl').open('a',encoding='utf-8') as out:
                    out.write(json.dumps({'config':c.to_dict(),'family':row['family'],'error':traceback.format_exc()})+'\n')
            status()
        study.analyze(rows)
        state['status']='completed_with_gaps' if state['failed'] or state['systems_failed'] or held else 'completed'
    except BaseException:
        state['status']='failed';state['error']=traceback.format_exc();raise
    finally:
        state['finished_utc']=datetime.now(timezone.utc).isoformat();status()


if __name__=='__main__':run()
