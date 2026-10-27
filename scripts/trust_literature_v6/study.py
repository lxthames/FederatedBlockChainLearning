"""Bounded literature pilot, immutable inputs, no automatic confirmation."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(name,'2')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from dataclasses import replace
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import traceback
import numpy as np
import psutil
import torch
from threadpoolctl import threadpool_limits
from .femnist import dump,digest
from .research.config import ExperimentConfig
from .research.runner_v2 import BenchmarkRunner
from .research.models import create_model
from .research.splits import make_splits
from .recovery import RecoveryPilot,detect_archive,recovery_binding
from .runtime import exclusive_lock

ROOT=Path(__file__).resolve().parents[2]
JOB=ROOT/'artifacts/literature-v6-pilot'
OUT=ROOT/'results/literature-v6-pilot'
DATA=ROOT/'data/trust-revision-v3-development/fashion_mnist-11'
METHODS=('fedavg','rank_trust','loss_quorum','flame','flshield_cluster','fldetector_pp')
RECOVERY=('retrain','fedrecover_partial','crab_fixedwindow')


def config(method,attack,rounds=30):
    return ExperimentConfig(experiment_id=f'v6-{method}-{attack}-s11',dataset='fashion_mnist',
        num_clients=50,participation_rate=.4,malicious_fraction=.3,dirichlet_alpha=.5,
        rounds=rounds,local_epochs=1,seed=11,attack=attack,attack_start_round=11,
        defense='two_layer' if method in ('rank_trust','loss_quorum') else method,
        batch_size=64,learning_rate=.01,optimizer_momentum=.9,device='cuda',num_classes=10,
        sampling_policy='bounded',vote_rule='rank_loss',candidate_clip_ratio=2,
        trust_reward=.02,rejection_trust_penalty=.02,
        ablation='validation_only' if method=='rank_trust' else 'full',
        recovery_signal='paired_loss_quorum' if method=='loss_quorum' else 'accuracy_then_loss',
        recovery_penalties_enabled=False,archive_updates=method=='fedavg')


def manifest():
    return [{'method':m,'config':config(m,a).to_dict()} for a in ('none','low_and_slow','model_replacement') for m in METHODS]


def hashes():
    return {str(p.relative_to(ROOT)):digest(p) for p in sorted(Path(__file__).parent.rglob('*.py'))}


def frozen_inputs():
    provenance=json.loads((DATA/'provenance.json').read_text())
    data={}
    for name,h in provenance['files'].items():
        if digest(DATA/name)!=h:raise ValueError('development data changed')
        data[str((DATA/name).relative_to(ROOT))]=h
    audit=ROOT/'docs/research/LITERATURE_V6_BASELINE_AUDIT.md'
    return {'source':hashes(),'data':data,'manifest':manifest(),
        'audit_sha256':digest(audit),'recovery_methods':list(RECOVERY),
        'primary_sources_sha256':digest(JOB/'sources/manifest.json'),
        'recovery_identity_roles':{'low_and_slow':['detector_selected'],
            'model_replacement':['detector_selected','oracle_diagnostic']},
        'policy':{'single_development_seed':11,'automatic_confirmation':False,
            'max_clean_loss':.05,'minimum_clean_utility':.20,
            'minimum_gradual_final_utility_gain':.01,'minimum_targeted_asr_reduction':.05,
            'oracle_rows_do_not_select_methods':True}}


def clean_gate(records,method):
    baseline=[r for r in records if r['method']=='fedavg' and r['attack']=='none']
    candidate=[r for r in records if r['method']==method and r['attack']=='none']
    if len(baseline)!=1 or len(candidate)!=1:return {'passed':False,'reason':'incomplete clean cell'}
    b,r=baseline[0],candidate[0]
    difference=r['final_utility']-b['final_utility']
    return {'passed':bool(b['final_utility']>=.20 and difference>=-.05 and r['unsupported_rounds']<=3
        and r['honest_validator_penalty_amount']==0), 'clean_utility_difference':difference,
        'baseline_learning_passed':b['final_utility']>=.20}


def enrich(folder,summary,method):
    events=[json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
    rounds=[e for e in events if e['event_type']=='round_completed']
    diagnostics=[e['payload'] for e in events if e['event_type']=='literature_defense']
    onset=summary['configuration']['attack_start_round']
    post=[e['payload'] for e in rounds if e['round_index']>=onset]
    targeted=[r['targeted_attack_success'] for r in post if r['targeted_attack_success'] is not None]
    clean_rates=summary.get('clean_trigger_prediction_history',[])
    trace=json.loads((folder/'chain-trace.json').read_text())
    malicious=set(summary['malicious_client_ids']);rejected=[];submitted=[]
    for e in rounds:
        p=e['payload'];submitted.extend(i for i in p['participants'] if i not in malicious)
        rejected.extend(i for i in p['participants'] if i not in malicious and i not in p['accepted_client_ids'])
    summary.update(method=method,post_onset_mean_asr=float(np.mean(targeted)) if targeted else None,
        post_onset_mean_clean_trigger_rate=float(np.mean(clean_rates[onset-1:])) if clean_rates else None,
        post_onset_mean_utility=float(np.mean([r['accuracy'] for r in post])),
        final_targeted_asr=targeted[-1] if targeted else None,
        honest_update_exclusion_rate=len(rejected)/max(1,len(submitted)),
        participants=[e['payload']['participants'] for e in rounds],
        quarantined=diagnostics[-1].get('quarantined',[]) if diagnostics else [],
        numerical_warning_count=sum(len(r.get('algorithm_warnings',[]))+len(r.get('covariance_warnings',[])) for r in diagnostics),
        pre_onset_model_hashes=[r['model_hash'] for r in trace if r.get('round',0)<onset],
        archive_file_bytes=sum(p.stat().st_size for p in (folder/'offchain-history').glob('*.pt')),
        evaluation_role='training-only development holdout; single seed',publication_ready=False)
    dump(folder/'summary.json',summary)
    return summary


def analyze(records,recoveries,state):
    comparisons=[]
    clean=next((r for r in records if r['method']=='fedavg' and r['attack']=='none'),None)
    for r in records:
        base=next((b for b in records if b['method']=='fedavg' and b['attack']==r['attack']),None)
        if base is None:continue
        matched=(r['participants']==base['participants'] and r['data_fingerprint']==base['data_fingerprint']
                 and r['test_fingerprint']==base['test_fingerprint'])
        targeted=r['attack']=='model_replacement'
        reduction=base['post_onset_mean_asr']-r['post_onset_mean_asr'] if targeted else None
        utility=r['final_utility']-base['final_utility']
        attack_valid=None
        if clean and r['attack']=='low_and_slow':attack_valid=clean['final_utility']-base['final_utility']>=.01
        elif clean and targeted:attack_valid=base['post_onset_mean_asr']-clean['post_onset_mean_clean_trigger_rate']>=.10
        promising=(r['attack']!='none' and r['method']!='fedavg' and clean_gate(records,r['method'])['passed']
            and matched and attack_valid and utility>=-.05 and
            (reduction>=.05 if targeted else utility>=.01))
        comparisons.append({'method':r['method'],'attack':r['attack'],'final_utility':r['final_utility'],
            'utility_effect_vs_fedavg':utility,'post_onset_asr':r['post_onset_mean_asr'],
            'asr_reduction_vs_fedavg':reduction,'attack_effective_against_fedavg':attack_valid,
            'matched_participants_and_data':matched,'honest_update_exclusion_rate':r['honest_update_exclusion_rate'],
            'recoveries':r['recovery_count'],'runtime_seconds':r['runtime_seconds_total'],
            'numerical_warning_count':r['numerical_warning_count'],
            'simulated_communication_bytes':r['transmitted_bytes'],'archive_file_bytes':r['archive_file_bytes'],
            'worth_replication_review':bool(promising)})
    complete=state['completed']==18 and state['failed']==0 and state['held']==0
    report={'status':state,'comparisons':comparisons,'recovery_diagnostics':recoveries,
        'publication_ready':False,'complete_training_grid':complete,
        'automatic_promotion':False,'limitation':'One reused development seed. Partial-participation detection and recovery adaptations are not certified reproductions. Oracle rows are conditional diagnostics only.',
        'clean_gates':{m:clean_gate(records,m) for m in METHODS}}
    dump(JOB/'analysis.json',report)
    lines=['# Literature pilot comparison','',report['limitation'],'',
        '| Method | Condition | Utility | ASR after onset | Honest exclusion | Runtime (min) | Review replication |',
        '|---|---|---:|---:|---:|---:|---|']
    for r in comparisons:
        asr='n/a' if r['post_onset_asr'] is None else f"{100*r['post_onset_asr']:.1f}%"
        lines.append(f"| {r['method']} | {r['attack']} | {100*r['final_utility']:.1f}% | {asr} | {100*r['honest_update_exclusion_rate']:.1f}% | {r['runtime_seconds']/60:.1f} | {r['worth_replication_review']} |")
    lines+=['','Recovery branches are conditional on their stated suspect sets; oracle rows cannot validate end-to-end detection.','',
        '| Recovery | Suspect source | Condition | Status | Utility | Exact / estimated requests |',
        '|---|---|---|---|---:|---:|']
    for r in recoveries:
        lines.append(f"| {r['method']} | {r['identity_role']} | {r['attack']} | {r['status']} | {r.get('final_utility','n/a')} | {r.get('exact_client_requests',0)} / {r.get('estimated_client_requests',0)} |")
    (JOB/'comparison.md').write_text('\n'.join(lines)+'\n')


def benchmark():
    """Two full-data rounds per new defense; no grid changes from the outcomes."""
    torch.set_num_threads(2)
    x,y,ex,ey=[np.load(DATA/f'{n}.npy',allow_pickle=False) for n in ('features','labels','development_features','development_labels')]
    records=[]
    source=hashes()
    snapshot=hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest()[:12]
    with threadpool_limits(limits=2):
        for method in ('fedavg','flame','flshield_cluster','fldetector_pp'):
            folder=ROOT/f'results/literature-v6-benchmark-{snapshot}'/method
            c=replace(config(method,'none',rounds=2),experiment_id=f'v6-benchmark-{method}',archive_updates=False)
            if (folder/'summary.json').exists():s=json.loads((folder/'summary.json').read_text())
            else:s=BenchmarkRunner(c,folder).run_arrays(x,y,lambda:create_model(c.dataset,x.shape[1:],10),(ex,ey))
            records.append({'method':method,'seconds_per_round':s['runtime_seconds_total']/2})
    forecast=sum(r['seconds_per_round']*30*3 for r in records)+2*3*30*max(r['seconds_per_round'] for r in records)
    dump(JOB/'benchmark.json',{'runs':records,'rough_training_grid_minutes':forecast/60,'source':source,
        'results_root':str(ROOT/f'results/literature-v6-benchmark-{snapshot}'),
        'note':'Approximation: rank methods cost bounded by slowest benchmark; recovery cost additional and depends on suspect sets. Shared GPU contention varies.'})


def run():
    with exclusive_lock(JOB/'coordinator.lock'),threadpool_limits(limits=2):
        state={'status':'starting','pid':os.getpid(),'completed':0,'failed':0,'held':0,
            'recovery_completed':0,'recovery_unsupported':0,'recovery_failed':0,'total':18,
            'maximum_recovery_branches':9,'publication_ready':False,'started_utc':datetime.now(timezone.utc).isoformat()}
        dump(JOB/'status.json',state)
        records=[];recoveries=[]
        try:
            if not torch.cuda.is_available():raise RuntimeError('CUDA is required')
            torch.set_num_threads(2)
            frozen=frozen_inputs();path=JOB/'frozen-inputs.json'
            if path.exists() and json.loads(path.read_text())!=frozen:raise ValueError('Frozen pilot inputs changed')
            dump(path,frozen)
            dump(JOB/'environment.json',{'python':platform.python_version(),'gpu':torch.cuda.get_device_name(0),
                'versions':{n:importlib.metadata.version(n) for n in ('torch','torchvision','numpy','scikit-learn','scipy','hdbscan')},
                'training_streams':'unchanged per-participant/round domains; separate defense RNG domains'})
            # Avoid a third concurrent training process; original study continues.
            prior=ROOT/'artifacts/loss-monitor-v5-pilot/status.json';deadline=time.monotonic()+4*3600
            while prior.exists():
                previous=json.loads(prior.read_text())
                if previous['status'] not in ('running','starting'):break
                if not psutil.pid_exists(previous['pid']):raise RuntimeError('V5 has stale running status; inspect before proceeding')
                if time.monotonic()>deadline:raise TimeoutError('V5 did not finish within four hours')
                state.update(status='waiting_for_v5',updated_utc=datetime.now(timezone.utc).isoformat());dump(JOB/'status.json',state)
                time.sleep(15)
            x,y,ex,ey=[np.load(DATA/f'{n}.npy',allow_pickle=False) for n in ('features','labels','development_features','development_labels')]
            split=make_splits(y,config('flshield_cluster','none'),external_test=True)
            counts=np.array([np.bincount(y[v],minlength=10) for v in split.validation])
            dump(JOB/'class-coverage.json',{'minimum_exclusive':10,
                'qualifying_validators_by_class':(counts>10).sum(0).tolist(),
                'median_validation_examples':float(np.median(counts.sum(1))),
                'split_sha256':hashlib.sha256(json.dumps(split.manifest(),sort_keys=True).encode()).hexdigest()})
            for row in frozen['manifest']:
                if hashes()!=frozen['source']:raise ValueError('Frozen source changed')
                method=row['method'];c=ExperimentConfig(**row['config'])
                state.update(status='running',current=c.experiment_id,updated_utc=datetime.now(timezone.utc).isoformat());dump(JOB/'status.json',state)
                if c.attack!='none' and not clean_gate(records,method)['passed']:
                    state['held']+=1;dump(OUT/c.experiment_id/'held.json',clean_gate(records,method));continue
                folder=OUT/c.experiment_id
                try:
                    if (folder/'summary.json').exists():
                        s=json.loads((folder/'summary.json').read_text());expected=hashlib.sha256(json.dumps(c.to_dict(),sort_keys=True).encode()).hexdigest()
                        if s.get('config_hash')!=expected or s.get('status')!='completed':raise ValueError('invalid completed output')
                    else:
                        if folder.exists() and any(folder.iterdir()):folder.rename(folder.with_name(folder.name+f'.interrupted-{time.time_ns()}'))
                        s=BenchmarkRunner(c,folder).run_arrays(x,y,lambda:create_model(c.dataset,x.shape[1:],10),(ex,ey))
                    records.append(enrich(folder,s,method));state['completed']+=1
                except Exception:
                    state['failed']+=1
                    with (JOB/'failures.jsonl').open('a') as f:f.write(json.dumps({'experiment_id':c.experiment_id,'error':traceback.format_exc()})+'\n')
                analyze(records,recoveries,state);dump(JOB/'status.json',state)
            for attack in ('low_and_slow','model_replacement'):
                source_config=config('fedavg',attack);source=OUT/source_config.experiment_id
                if not (source/'summary.json').exists():
                    state['recovery_unsupported']+=3 if attack=='low_and_slow' else 6;continue
                if hashes()!=frozen['source']:raise ValueError('Frozen source changed')
                suspects,reports=detect_archive(source,lambda:create_model('fashion_mnist',x.shape[1:],10),11,30)
                dump(JOB/f'detected-{attack}.json',{'suspects':suspects,'reports':reports,'source_trace_sha256':digest(source/'chain-trace.json')})
                oracle=json.loads((source/'summary.json').read_text())['malicious_client_ids']
                roles=[('detector_selected',suspects)]+([('oracle_diagnostic',oracle)] if attack=='model_replacement' else [])
                for role,identities in roles:
                    for method in RECOVERY:
                        if hashes()!=frozen['source']:raise ValueError('Frozen source changed')
                        c=replace(source_config,experiment_id=f'v6-recovery-{method}-{role}-{attack}-s11',archive_updates=False)
                        folder=OUT/c.experiment_id
                        state.update(status='recovery_diagnostic',current=c.experiment_id,updated_utc=datetime.now(timezone.utc).isoformat());dump(JOB/'status.json',state)
                        try:
                            if (folder/'summary.json').exists():
                                binding=json.loads((folder/'configuration.json').read_text())
                                if binding!=recovery_binding(source,c,method,identities,role):raise ValueError('stale recovery cache')
                                s=json.loads((folder/'summary.json').read_text())
                                if s.get('status') not in {'completed','unsupported_no_detected_suspects'}:raise ValueError('incomplete recovery cache')
                            else:
                                if folder.exists() and any(folder.iterdir()):folder.rename(folder.with_name(folder.name+f'.interrupted-{time.time_ns()}'))
                                s=RecoveryPilot(c,folder).recover(source,method,identities,role,x,y,ex,ey,lambda:create_model('fashion_mnist',x.shape[1:],10))
                            s['attack']=attack;dump(folder/'summary.json',s);recoveries.append(s)
                            state['recovery_completed' if s['status']=='completed' else 'recovery_unsupported']+=1
                        except Exception:
                            state['recovery_failed']+=1
                            with (JOB/'failures.jsonl').open('a') as f:f.write(json.dumps({'experiment_id':c.experiment_id,'error':traceback.format_exc()})+'\n')
                        analyze(records,recoveries,state);dump(JOB/'status.json',state)
            state['status']='completed_with_gaps' if state['failed'] or state['held'] or state['recovery_failed'] else 'completed'
        except BaseException:
            state.update(status='failed',error=traceback.format_exc());raise
        finally:
            state['finished_utc']=datetime.now(timezone.utc).isoformat();dump(JOB/'status.json',state)
            analyze(records,recoveries,state)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--benchmark',action='store_true');args=parser.parse_args()
    benchmark() if args.benchmark else run()
