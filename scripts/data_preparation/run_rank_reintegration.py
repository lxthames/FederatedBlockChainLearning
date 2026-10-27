"""Bounded reintegration of rank-loss voting, using completed matched controls.

Nine existing rank-only cells are reused by exact config/source/data checks.
At most 27 additional development cells run; no confirmation launches.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import traceback
import numpy as np
import torch
from scripts.trust_math_lab import study as parent
from scripts.trust_math_lab.femnist import dump, digest
from scripts.trust_math_lab.research.config import ExperimentConfig
from scripts.trust_math_lab.research.models import create_model
from scripts.trust_math_lab.research.runner_v2 import BenchmarkRunner

ROOT = Path(__file__).resolve().parents[2]
JOB = ROOT/'artifacts/rank-reintegration-v1'
OUT = ROOT/'results/rank-reintegration-v1'
SEEDS = (11,17,23)
VARIANTS = ('trust_only','recovery_only','combined')
ATTACKS = ('none','low_and_slow','model_replacement')


def fingerprint_config(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()


def config(variant,seed,attack):
    c = parent.make_config('rank_loss',seed,attack)
    changes = dict(experiment_id=f'rank-reintegration-{variant}-{attack}-s{seed}')
    if variant=='trust_only':
        changes.update(ablation='validation_only',trust_reward=.02,rejection_trust_penalty=.02)
    elif variant=='recovery_only':
        changes.update(ablation='no_penalties',trust_reward=0,rejection_trust_penalty=0)
    elif variant=='combined':
        changes.update(ablation='full',trust_reward=.02,rejection_trust_penalty=.02)
    else:
        raise ValueError('unknown reintegration variant')
    return replace(c,**changes)


def manifest():
    return [{'stage':'clean' if attack=='none' else 'attacked','variant':variant,
             'config':config(variant,seed,attack).to_dict()}
            for attack in ATTACKS for variant in VARIANTS for seed in SEEDS]


def validate_reference(summary,expected):
    if summary.get('status')!='completed' or summary.get('configuration')!=expected:
        raise ValueError('Reference configuration/status mismatch')
    if summary.get('config_hash')!=fingerprint_config(expected):
        raise ValueError('Reference configuration fingerprint mismatch')


def complete(records,variant,attack):
    group=[r for r in records if r['variant']==variant and r['attack']==attack]
    if sorted(r['seed'] for r in group)!=list(SEEDS):return None
    return {r['seed']:r for r in group}


def clean_gate(records,references,variant):
    group=complete(records,variant,'none')
    base=complete(references,'rank_only','none')
    if group is None or base is None:return {'passed':False,'reason':'incomplete clean seed group'}
    differences=[group[s]['final_utility']-base[s]['final_utility'] for s in SEEDS]
    utility_ok=bool(np.mean(differences)>=-.05 and min(differences)>=-.10)
    availability=all(group[s]['unsupported_rounds']<=max(2,int(.1*group[s]['rounds_completed'])) for s in SEEDS)
    # Development screen: cap clean economic damage at 5% of initial aggregate
    # stake. This is a disclosed engineering criterion, not an economic theorem.
    penalties=all(group[s]['honest_validator_penalty_amount']<=.05*group[s]['num_clients']*group[s]['configuration']['initial_stake'] for s in SEEDS)
    return {'passed':utility_ok and availability and penalties,
            'paired_clean_utility_effects':differences,'utility_passed':utility_ok,
            'availability_passed':availability,'honest_penalty_budget_passed':penalties}


def analyze(records,references):
    comparisons=[]
    metrics=('final_utility','recovery_count','wrongful_slashing_count','honest_validator_penalty_amount',
             'unsupported_rounds','monitoring_quorum_failures','detection_false_positive_rate',
             'detection_false_negative_rate','detection_delay_rounds','mean_targeted_attack_success',
             'recovery_utility_gain_sum','recovery_regret_sum','runtime_seconds_total')
    for variant in VARIANTS:
        for attack in ATTACKS:
            group=complete(records,variant,attack);base=complete(references,'rank_only',attack)
            if group is None or base is None:continue
            entry={'variant':variant,'attack':attack,'seeds':list(SEEDS),'metrics':{},
                   'paired_final_utility_effects':[group[s]['final_utility']-base[s]['final_utility'] for s in SEEDS]}
            for metric in metrics:
                values=[group[s].get(metric) for s in SEEDS]
                if all(isinstance(v,(int,float)) for v in values):
                    entry['metrics'][metric]={'seed_values':values,'mean':float(np.mean(values))}
            for label,runs in [('rank_only',base),('reintegrated',group)]:
                entry[label+'_eligibility_mean']=[float(np.mean(list(runs[s]['client_eligible_round_counts'].values()))/runs[s]['rounds_completed']) for s in SEEDS]
                entry[label+'_accepted_fraction']=[1-sum(runs[s]['client_rejection_counts'].values())/max(1,sum(runs[s]['client_submission_counts'].values())) for s in SEEDS]
            if attack=='model_replacement':
                entry['paired_mean_asr_reductions']=[base[s]['mean_targeted_attack_success']-group[s]['mean_targeted_attack_success'] for s in SEEDS]
            comparisons.append(entry)
    dump(JOB/'analysis.json',{'clean_gates':{v:clean_gate(records,references,v) for v in VARIANTS},
        'complete_comparisons':comparisons,'publication_ready':False,
        'limitations':'Three reused development seeds/writer sets; exploratory paired effects, no significance or security guarantee. Clean screens gate attacked work; recovery may still leave strong targeted attacks.'})


def replay(c,output,summary):
    from scripts.trust_math_lab.research.evm import EVMAdapter
    trace=json.loads((output/'chain-trace.json').read_text())
    folder=output/'evm-replay'
    if (folder/'agreement.json').exists():return
    if folder.exists() and any(folder.iterdir()):folder.rename(folder.with_name(folder.name+f'.interrupted-{time.time_ns()}'))
    adapter=EVMAdapter(replace(c,experiment_id=c.experiment_id+'-replay'),folder)
    adapter.genesis(trace[0])
    for record in trace[1:]:adapter.round(record)
    if adapter.contract.functions.activeVersion().call()!=summary['active_version']:
        raise AssertionError('Final active model mismatch')
    for client,stake in summary['final_stakes'].items():
        if adapter.contract.functions.stake(adapter.account(int(client))).call()!=adapter.units(stake):
            raise AssertionError('Final stake mismatch')
    dump(folder/'agreement.json',{'status':'passed','rounds':len(trace)-1,'stake_agreement':True,
        'active_version':summary['active_version'],'trace_sha256':digest(output/'chain-trace.json'),
        'transaction_count':len(adapter.receipts),'gas_used':sum(r['gas_used'] for r in adapter.receipts)})


def run():
    JOB.mkdir(parents=True,exist_ok=True)
    state={'status':'starting','pid':os.getpid(),'completed':0,'failed':0,'held':0,'systems_failed':0,
           'reused_reference_runs':0,'new_runs_maximum':27,'publication_ready':False,
           'started_utc':datetime.now(timezone.utc).isoformat()}
    dump(JOB/'status.json',state)
    try:
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        torch.set_num_threads(2)
        prerequisite=json.loads((parent.JOB/'assessment.json').read_text())
        if not prerequisite['methods']['rank_loss']['ready_for_reintegration_review']:
            raise RuntimeError('Rank-loss prerequisite assessment did not pass')
        parent_frozen=json.loads((parent.JOB/'frozen-inputs.json').read_text())
        if parent.source_hashes()!=parent_frozen['source']:raise RuntimeError('Parent source changed')
        for name,h in parent_frozen['data'].items():
            if digest(parent.DATA/name)!=h:raise RuntimeError('Parent development data changed')
        references=[];reference_hashes={}
        for attack in ATTACKS:
            for seed in SEEDS:
                c=parent.make_config('rank_loss',seed,attack)
                path=parent.OUT/c.experiment_id/'summary.json'
                summary=json.loads(path.read_text());validate_reference(summary,c.to_dict())
                summary=dict(summary,variant='rank_only',reference_path=str(path.relative_to(ROOT)))
                references.append(summary);reference_hashes[str(path.relative_to(ROOT))]=digest(path)
        state['reused_reference_runs']=len(references)
        rows=manifest()
        source=dict(parent.source_hashes());source[str(Path(__file__).relative_to(ROOT))]=digest(__file__)
        frozen={'source':source,'references':reference_hashes,'manifest':rows,'data':parent_frozen['data'],
                'prerequisite_assessment_sha256':digest(parent.JOB/'assessment.json')}
        frozen_path=JOB/'frozen-inputs.json'
        if frozen_path.exists() and json.loads(frozen_path.read_text())!=frozen:raise RuntimeError('Reintegration inputs changed')
        dump(frozen_path,frozen)
        dump(JOB/'environment.json',{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,
            'gpu':torch.cuda.get_device_name(0),'concurrent_workloads':True})
        median=statistics.median(r['runtime_seconds_total'] for r in references)
        dump(JOB/'runtime-forecast.json',{'reference_median_seconds':median,
            'nominal_new_training_minutes':median*27/60,
            'caveat':'Planning extrapolation from same-rank 40-round runs; skips reduce work; EVM replay and shared-host contention add time.'})
        records=[];held=[];replayed=False
        for row in rows:
            if any(digest(ROOT/n)!=h for n,h in source.items()):raise RuntimeError('Frozen source changed')
            c=ExperimentConfig(**row['config']);v=row['variant']
            state.update(status='running',stage=row['stage'],current=c.experiment_id,
                         updated_utc=datetime.now(timezone.utc).isoformat());dump(JOB/'status.json',state)
            if row['stage']=='attacked':
                gate=clean_gate(records,references,v)
                if not gate['passed']:
                    held.append({'experiment_id':c.experiment_id,'clean_gate':gate});state['held']+=1
                    dump(JOB/'held.json',held);dump(JOB/'status.json',state);continue
            output=OUT/c.experiment_id
            try:
                if (output/'summary.json').exists():
                    summary=json.loads((output/'summary.json').read_text());validate_reference(summary,c.to_dict())
                else:
                    if output.exists() and any(output.iterdir()):output.rename(output.with_name(output.name+f'.interrupted-{time.time_ns()}'))
                    folder=parent.DATA/f'femnist-{c.seed}'
                    x,y,ex,ey=[np.load(folder/f'{n}.npy',allow_pickle=False) for n in ('features','labels','development_features','development_labels')]
                    summary=BenchmarkRunner(c,output).run_arrays(x,y,lambda:create_model(c.dataset,x.shape[1:],62),(ex,ey))
                summary=dict(summary,variant=v,evaluation_role='training-only development holdout')
                dump(output/'summary.json',summary);records.append(summary);state['completed']+=1
                analyze(records,references)
                # One real-data full trace checks the same decisions in Solidity.
                if v=='combined' and not replayed:
                    try:
                        replay(c,output,summary);replayed=True
                    except Exception:
                        state['systems_failed']+=1
                        with (JOB/'systems-failures.jsonl').open('a',encoding='utf-8') as f:
                            f.write(json.dumps({'experiment_id':c.experiment_id,'error':traceback.format_exc()})+'\n')
            except Exception:
                state['failed']+=1
                with (JOB/'failures.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps({'config':c.to_dict(),'error':traceback.format_exc()})+'\n')
            dump(JOB/'status.json',state)
        analyze(records,references)
        state['status']='completed_with_gaps' if state['failed'] or state['held'] or state['systems_failed'] else 'completed'
    except BaseException:
        state.update(status='failed',error=traceback.format_exc());raise
    finally:
        state['finished_utc']=datetime.now(timezone.utc).isoformat();dump(JOB/'status.json',state)


if __name__=='__main__':run()
