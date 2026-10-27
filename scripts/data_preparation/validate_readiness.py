"""Bounded development audit; never launches confirmatory experiments."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback
import numpy as np
import torch
from scripts.trust_extension import study
from scripts.trust_extension.femnist import dump, digest
from scripts.trust_extension.research.config import ExperimentConfig

ROOT=Path(__file__).resolve().parents[2]
JOB=ROOT/'artifacts/readiness-validation-v1'
OUT=ROOT/'results/readiness-validation-v1'

def main():
    JOB.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    torch.set_num_threads(2)
    rows=[]
    # Prespecified small grid, all development seeds; no confirmatory seed access.
    for seed in (3,5,7):
        for lr in (0.01,0.05,0.1):
            row=study.cell('development','femnist','fedavg','none',seed,rounds=100,
                           device='cuda',learning_rate=lr)
            rows.append(('femnist-convergence',row))
        for name,changes in (
            ('validation-only',dict(ablation='validation_only')),
            ('no-trust-exclusion',dict(minimum_trust=0.0,trust_floor=0.0)),
            ('permissive-votes',dict(minimum_validation_delta=-1e12,minimum_trust=0.0,trust_floor=0.0))):
            row=study.cell('development','fashion_mnist','full','none',seed,rounds=20,
                           device='cuda',**changes)
            rows.append((name,row))
    frozen={'source':study.code_hashes(),'launcher':digest(__file__),'rows':rows}
    path=JOB/'frozen-inputs.json'
    canonical=json.loads(json.dumps(frozen))
    if path.exists() and json.loads(path.read_text())!=canonical:
        raise RuntimeError('Audit inputs changed; use new version')
    dump(path,frozen)
    state={'status':'running','pid':os.getpid(),'completed':0,'failed':0,'total':len(rows),
           'publication_ready':False,'started_utc':datetime.now(timezone.utc).isoformat()}
    results=[]
    for label,row in rows:
        if study.code_hashes()!=frozen['source']: raise RuntimeError('Source changed')
        c=ExperimentConfig(**row['config'])
        c=replace(c,experiment_id='audit-'+label+'-'+c.experiment_id)
        state.update(current=c.experiment_id,updated_utc=datetime.now(timezone.utc).isoformat())
        dump(JOB/'status.json',state)
        try:
            s=study.execute(c,OUT/c.experiment_id)
            record={k:s.get(k) for k in ('seed','dataset','final_utility','accuracy_history',
                'recovery_count','unsupported_rounds','runtime_seconds_total',
                'client_rejection_counts','client_submission_counts','client_eligible_round_counts')}
            record.update(diagnostic=label,learning_rate=c.learning_rate)
            if c.dataset=='femnist':
                x,y,test=study.load(c)
                _,counts=np.unique(test[1],return_counts=True)
                record['test_majority_class_accuracy']=float(max(counts)/sum(counts))
                split=json.loads(Path(c.natural_splits_path).read_text())
                record['train_samples_per_client']=[len(v) for v in split['train']]
            results.append(record);state['completed']+=1
        except Exception:
            state['failed']+=1
            results.append({'diagnostic':label,'config':c.to_dict(),'error':traceback.format_exc()})
        dump(JOB/'diagnostics.json',results)
        dump(JOB/'status.json',state)
    # Compute the original development gates explicitly, including held attacks.
    prior=[json.loads(p.read_text()) for p in (ROOT/'results/trust-extension-v1-gpu/development').glob('*/summary.json')]
    gates={f'{d}:{a}':study.gate(prior,d,a) for d in ('fashion_mnist','femnist')
           for a in ('none','model_replacement','low_and_slow')}
    dump(JOB/'original-development-gates.json',gates)
    state.update(status='completed_with_failures' if state['failed'] else 'completed',
                 finished_utc=datetime.now(timezone.utc).isoformat(),
                 next_step='Review diagnostics; validate attacks on any revised configuration before freezing final study. No automatic confirmation launch.')
    dump(JOB/'status.json',state)

if __name__=='__main__': main()
