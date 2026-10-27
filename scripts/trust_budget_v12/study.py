"""Bounded clean development interventions: training budget, CNN width and mass."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(name,'2')

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
import traceback
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .pairing import compare_pair, digest_value, TRAINING_FIELDS
from .research.config import ExperimentConfig
from .research.models import create_model
from .research.runner_v2 import BenchmarkRunner
from .runtime import exclusive_lock

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data/trust-revision-v3-development'
JOB=ROOT/'artifacts/budget-v12'
OUT=ROOT/'results/budget-v12'
PARENT=ROOT/'artifacts/screen-v11'
POLICY=ROOT/'docs/research/BUDGET_V12.md'
SEEDS=(11,17,23)
MODELS=('default','wide_cnn')
BUDGETS=('epoch','fixed_steps')
ARMS=('fedavg','clip_only','scaled_nominal','scaled_renormalized','clip_damping_replay')
FILES=('config.json','summary.json','splits.json','events.jsonl','chain-trace.json','diagnostic-inputs.json')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)


def config(model,budget,arm,seed):
    if model not in MODELS or budget not in BUDGETS or arm not in ARMS or seed not in SEEDS:
        raise ValueError('Unknown V12 cell')
    scaled=arm.startswith('scaled_')
    return ExperimentConfig(experiment_id=f'v12-femnist-{model}-{budget}-{arm}-s{seed}',
        dataset='femnist',num_clients=50,participation_rate=.4,malicious_fraction=.3,
        seed=seed,rounds=40,attack='none',attack_start_round=11,num_classes=62,
        batch_size=16,learning_rate=.05,optimizer_momentum=.9,device='cuda',
        sampling_policy='bounded',model=model,local_training_budget=budget,local_steps=4,
        defense='fedavg' if arm=='fedavg' else 'two_layer',
        vote_rule='scaled_loss' if scaled else 'rank_loss',ablation='validation_only',
        candidate_clip_ratio=2.,natural_splits_path=str(DATA/f'femnist-{seed}/natural-splits.json'),
        validation_distribution='private',trust_updates='fixed',trust_weighting='fixed',trust_eligibility='fixed',
        trust_reward=0.,rejection_trust_penalty=0.,rejection_trust_policy='none',
        ranking_keep_fraction=.8 if scaled else 1.,loss_tolerance=0.,loss_standard_errors=2.,
        aggregation_mass_policy='nominal' if arm=='scaled_nominal' else ('replay' if arm=='clip_damping_replay' else 'renormalized'))


def manifest():
    # Finish original CNN before the wider backbone; sources precede dependencies.
    return [{'model':m,'budget':b,'arm':a,'seed':s,'config':config(m,b,a,s).to_dict(),
             'dependency':config(m,b,'scaled_nominal',s).experiment_id if a=='clip_damping_replay' else None}
            for m in MODELS for b in BUDGETS for s in SEEDS for a in ARMS]


def inputs():
    data={}
    for seed in SEEDS:
        folder=DATA/f'femnist-{seed}';provenance=read(folder/'provenance.json')
        for name in ('features.npy','labels.npy','development_features.npy','development_labels.npy','natural-splits.json'):
            actual=digest(folder/name)
            if provenance['files'].get(name)!=actual:raise ValueError('Prepared data changed')
            data[str((folder/name).relative_to(ROOT))]=actual
        data[str((folder/'provenance.json').relative_to(ROOT))]=digest(folder/'provenance.json')
    return {'source':{str(p.relative_to(ROOT)):digest(p) for p in sorted(Path(__file__).parent.rglob('*.py'))},
        'data':data,'manifest':manifest(),'policy_sha256':digest(POLICY),
        'parent_freeze_sha256':digest(PARENT/'frozen-inputs.json'),
        'evaluation_role':'search-exposed clean development diagnosis',
        'automatic_confirmation':False,'publication_ready':False}


def freeze():
    value=inputs();path=JOB/'frozen-inputs.json'
    if path.exists() and read(path)!=value:raise ValueError('Frozen V12 changed')
    dump(path,value)
    return value


def seal(folder):dump(folder/'completion.json',{name:digest(folder/name) for name in FILES})


def load_completed(folder,c,binding):
    if read(folder/'completion.json')!={name:digest(folder/name) for name in FILES}:
        raise ValueError('Completed artifact digest mismatch')
    summary=read(folder/'summary.json')
    expected=hashlib.sha256(json.dumps(c.to_dict(),sort_keys=True).encode()).hexdigest()
    if summary.get('config_hash')!=expected or summary.get('status')!='completed':
        raise ValueError('Completed configuration/status mismatch')
    if summary.get('diagnostic_input_binding')!=binding or read(folder/'diagnostic-inputs.json')!=binding:
        raise ValueError('Diagnostic input binding mismatch')
    return summary


def replay_inputs(row):
    if row['arm']!='clip_damping_replay':return {'mass_schedule':None,'source':None}
    source=OUT/row['dependency']
    c=config(row['model'],row['budget'],'scaled_nominal',row['seed'])
    load_completed(source,c,{'mass_schedule':None,'source':None})
    split=read(source/'splits.json')
    rounds=[json.loads(line) for line in (source/'events.jsonl').read_text().splitlines()]
    rounds=[e for e in rounds if e['event_type']=='round_completed']
    if [r['round_index'] for r in rounds]!=list(range(1,c.rounds+1)):
        raise ValueError('Incomplete mass source')
    schedule=[]
    for e in rounds:
        p=e['payload'];offered=p['participants'];accepted=p['accepted_client_ids']
        if not set(accepted)<=set(offered):raise ValueError('Source has unoffered accepted clients')
        schedule.append(sum(len(split['train'][i]) for i in accepted)/sum(len(split['train'][i]) for i in offered))
    return {'mass_schedule':schedule,'source':{'experiment_id':c.experiment_id,
        'completion_sha256':digest(source/'completion.json'),'schedule_sha256':digest_value(schedule)}}


def compare_intervention(left,right,allowed=()):
    """Allow only a prespecified training change; verify stored bindings first."""
    allowed=set(allowed)
    if not allowed<= {'model','local_training_budget'}:raise ValueError('Unapproved intervention')
    a,b=left['configuration'],right['configuration']
    reasons=[]
    for row in (left,right):
        if row.get('training_binding')!=digest_value({k:row['configuration'][k] for k in TRAINING_FIELDS}):
            reasons.append('invalid_training_binding')
    actual={k for k in TRAINING_FIELDS if a[k]!=b[k]}
    if actual!=allowed:reasons.append('unexpected_training_intervention')
    l,r=dict(left),dict(right)
    l['training_binding']=digest_value({k:a[k] for k in TRAINING_FIELDS if k not in allowed})
    r['training_binding']=digest_value({k:b[k] for k in TRAINING_FIELDS if k not in allowed})
    if 'model' in allowed:
        if {a['model'],b['model']}!={'default','wide_cnn'}:reasons.append('unexpected_backbone')
        if not left.get('initial_model_hash') or not right.get('initial_model_hash'):reasons.append('missing_initialization')
        l['initial_model_hash']=r['initial_model_hash']='declared model-width intervention'
    paired=compare_pair(l,r)
    return {'matched':not reasons and paired['matched'],'mismatches':reasons+paired['mismatches'],
            'allowed_training_changes':sorted(allowed)}


def trace_metrics(folder,summary):
    events=[json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
    local=[e['payload'] for e in events if e['event_type']=='local_update_diagnostic']
    rounds=[e['payload'] for e in events if e['event_type']=='round_completed']
    split=read(folder/'splits.json')
    assert sum(e['optimizer_steps'] for e in local)==summary['local_training_work']['optimizer_steps']
    assert sum(e['processed_examples'] for e in local)==summary['local_training_work']['processed_examples']
    mass=[sum(len(split['train'][i]) for i in p['accepted_client_ids'])/
          sum(len(split['train'][i]) for i in p['participants']) for p in rounds]
    c=summary['configuration'];policy=c['aggregation_mass_policy']
    factors=summary['diagnostic_input_binding']['mass_schedule'] if policy=='replay' else (mass if policy=='nominal' else [1. if p['accepted_client_ids'] else 0. for p in rounds])
    counts=np.array([e['sample_count'] for e in local]);norms=np.array([e['unclipped_norm'] for e in local])
    get=lambda mapping,i:mapping.get(str(i),mapping.get(i,0))
    clients=[{'client':i,'size':len(ids),'submissions':get(summary['client_submission_counts'],i),
              'rejections':get(summary['client_rejection_counts'],i)} for i,ids in enumerate(split['train'])
             if get(summary['client_submission_counts'],i)>0]
    sizes=np.array([r['size'] for r in clients])
    rejection_rates=np.array([r['rejections']/r['submissions'] for r in clients])
    ordered=sorted(clients,key=lambda r:r['size'])
    rate=lambda rows:sum(r['rejections'] for r in rows)/sum(r['submissions'] for r in rows)
    return {'mean_accepted_nominal_mass':float(np.mean(mass)), 'mean_applied_coefficient_sum':float(np.mean(factors)),
            'size_norm_correlation':float(np.corrcoef(counts,norms)[0,1]) if np.std(counts)>0 and np.std(norms)>0 else None,
            'size_rejection_correlation':float(np.corrcoef(sizes,rejection_rates)[0,1]) if np.std(sizes)>0 and np.std(rejection_rates)>0 else None,
            'smallest_ten_rejection_rate':rate(ordered[:10]),'largest_ten_rejection_rate':rate(ordered[-10:])}


def analyze(records,state):
    lookup={(r['model'],r['budget'],r['arm'],r['seed']):r for r in records}
    comparisons=[]
    for row in records:
        m,b,a,s=row['model'],row['budget'],row['arm'],row['seed']
        candidates=[('vs_fedavg',(m,b,'fedavg',s),())]
        if b=='fixed_steps':candidates.append(('fixed_vs_epoch',(m,'epoch',a,s),('local_training_budget',)))
        if m=='wide_cnn':candidates.append(('wide_vs_default',('default',b,a,s),('model',)))
        if a=='scaled_renormalized':candidates.append(('renormalized_vs_nominal',(m,b,'scaled_nominal',s),()))
        if a=='clip_damping_replay':candidates.append(('damping_vs_clipping',(m,b,'clip_only',s),()))
        for kind,key,allowed in candidates:
            if key not in lookup:continue
            reference=lookup[key];pair=compare_intervention(row,reference,allowed)
            comparisons.append({'kind':kind,'model':m,'budget':b,'arm':a,'seed':s,**pair,
                'accuracy_difference':row['final_accuracy']-reference['final_accuracy'] if pair['matched'] else None,
                'reference_accuracy':reference['final_accuracy'],
                'candidate_unsupported_rounds':row['unsupported_rounds'],
                'candidate_honest_penalty':row['honest_validator_penalty_amount']})
    groups=[]
    for m in MODELS:
        for b in BUDGETS:
            for a in ARMS:
                rows=[lookup[(m,b,a,s)] for s in SEEDS if (m,b,a,s) in lookup]
                checks=[c for c in comparisons if c['kind']=='vs_fedavg' and (c['model'],c['budget'],c['arm'])==(m,b,a)]
                complete=len(rows)==3 and len(checks)==3 and all(c['matched'] for c in checks)
                mean=lambda key:float(np.mean([r[key] for r in rows])) if complete else None
                groups.append({'model':m,'budget':b,'arm':a,'complete':complete,
                    'mean_accuracy':mean('final_accuracy'),'mean_benign_exclusion':mean('benign_update_exclusion_rate'),
                    'mean_training_steps':float(np.mean([r['local_training_work']['optimizer_steps'] for r in rows])) if complete else None,
                    'clean_screen_passed':all(c['accuracy_difference']>=-.05 and c['reference_accuracy']>=.2
                        and c['candidate_unsupported_rounds']<=3 and c['candidate_honest_penalty']==0
                        for c in checks) if complete else None})
    interactions=[]
    for b in BUDGETS:
        for a in ARMS:
            differences=[]
            for seed in SEEDS:
                keys=[(m,b,arm,seed) for m in MODELS for arm in (a,'fedavg')]
                if not all(k in lookup for k in keys):continue
                wide,wide_base,original,original_base=[lookup[k] for k in
                    [('wide_cnn',b,a,seed),('wide_cnn',b,'fedavg',seed),('default',b,a,seed),('default',b,'fedavg',seed)]]
                checks=[compare_intervention(wide,wide_base),compare_intervention(original,original_base),
                        compare_intervention(wide,original,('model',)),compare_intervention(wide_base,original_base,('model',))]
                if all(c['matched'] for c in checks):
                    differences.append({'seed':seed,'defense_gap_change':
                        (wide['final_accuracy']-wide_base['final_accuracy'])-(original['final_accuracy']-original_base['final_accuracy'])})
            interactions.append({'kind':'width_by_defense','budget':b,'arm':a,'seeds':differences,
                'mean_gap_change':float(np.mean([r['defense_gap_change'] for r in differences])) if len(differences)==3 else None})
    for m in MODELS:
        for a in ARMS:
            differences=[]
            for seed in SEEDS:
                keys=[(m,b,arm,seed) for b in BUDGETS for arm in (a,'fedavg')]
                if not all(k in lookup for k in keys):continue
                fixed,fixed_base,epoch,epoch_base=[lookup[k] for k in
                    [(m,'fixed_steps',a,seed),(m,'fixed_steps','fedavg',seed),(m,'epoch',a,seed),(m,'epoch','fedavg',seed)]]
                checks=[compare_intervention(fixed,fixed_base),compare_intervention(epoch,epoch_base),
                        compare_intervention(fixed,epoch,('local_training_budget',)),
                        compare_intervention(fixed_base,epoch_base,('local_training_budget',))]
                if all(c['matched'] for c in checks):
                    differences.append({'seed':seed,'defense_gap_change':
                        (fixed['final_accuracy']-fixed_base['final_accuracy'])-(epoch['final_accuracy']-epoch_base['final_accuracy'])})
            interactions.append({'kind':'budget_by_defense','model':m,'arm':a,'seeds':differences,
                'mean_gap_change':float(np.mean([r['defense_gap_change'] for r in differences])) if len(differences)==3 else None})
    compact=[{k:r[k] for k in ('experiment_id','model','budget','arm','seed','final_accuracy','lower_decile_client_utility',
        'benign_update_exclusion_rate','unsupported_rounds','local_training_work','validation_work','runtime_seconds_total',
        'diagnostic_input_binding','trace_metrics')} for r in records]
    dump(JOB/'analysis.json',{'status':state,'runs':compact,'groups':groups,'comparisons':comparisons,'interactions':interactions,
        'automatic_promotion':False,'publication_ready':False})
    lines=['# V12 clean training-budget and architecture diagnostic','',
        '| Backbone | Local budget | Arm | Mean accuracy | Clean screen |','|---|---|---|---:|---|']
    for g in groups:
        value=f"{100*g['mean_accuracy']:.2f}%" if g['complete'] else 'incomplete / held'
        lines.append(f"| {g['model']} | {g['budget']} | {g['arm']} | {value} | {g['clean_screen_passed']} |")
    lines+=['','Means require all three matched seeds. Reused development seeds; no significance, security or novelty claim.',
        'Four steps changes processed-example budgets. Replay damping uses an exogenous recorded schedule and is a diagnostic control.',
        'Five-point clean screen is descriptive only. There is no automatic attack or confirmatory launch.','']
    (JOB/'comparison.md').write_text('\n'.join(lines),encoding='utf-8')


def parent_ready():
    state=read(PARENT/'status.json')
    if state['status'] in {'running','queued'}:return False
    if state['status'] not in {'completed','completed_with_gaps'} or state['failed'] or state['completed']+state['held']!=state['total']:
        raise RuntimeError('V11 ended unsuccessfully; inspect it before launching V12')
    try:
        with exclusive_lock(PARENT/'coordinator.lock'):pass
    except RuntimeError as error:
        if str(error)=='Coordinator already holds the experiment lock':return False
        raise
    return True


def run(wait_for_parent=False):
    with exclusive_lock(JOB/'coordinator.lock'),threadpool_limits(limits=2):
        frozen=read(JOB/'frozen-inputs.json')
        if inputs()!=frozen:raise ValueError('Frozen V12 changed')
        state={'status':'queued','pid':os.getpid(),'completed':0,'failed':0,'held':0,'total':len(frozen['manifest']),
               'publication_ready':False,'queued_utc':datetime.now(timezone.utc).isoformat()}
        records=[];dump(JOB/'status.json',state)
        try:
            while not parent_ready():
                if not wait_for_parent:raise RuntimeError('V11 is still running; use --wait-for-v11')
                state['waiting_for']='screen-v11';dump(JOB/'status.json',state);time.sleep(10)
            if inputs()!=frozen:raise ValueError('Frozen inputs changed while queued')
            if not torch.cuda.is_available():raise RuntimeError('CUDA required for V12')
            torch.set_num_threads(2)
            state.update(status='running',waiting_for=None,started_utc=datetime.now(timezone.utc).isoformat())
            dump(JOB/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,
                 'numpy':np.__version__,'gpu':torch.cuda.get_device_name(0),'parent_final_status':read(PARENT/'status.json')})
            dump(JOB/'status.json',state)
            for row in frozen['manifest']:
                if inputs()!=frozen:raise ValueError('Inputs changed during V12')
                c=ExperimentConfig(**row['config']);folder=OUT/c.experiment_id
                state['current']=c.experiment_id;dump(JOB/'status.json',state)
                try:binding=replay_inputs(row)
                except (FileNotFoundError,ValueError) as error:
                    if row['dependency'] is None:raise
                    dump(folder/'held.json',{'dependency':row['dependency'],'reason':str(error)})
                    state['held']+=1;analyze(records,state);dump(JOB/'status.json',state);continue
                try:
                    if (folder/'completion.json').exists():summary=load_completed(folder,c,binding)
                    else:
                        if folder.exists():
                            target=folder.with_name(folder.name+f'.interrupted-{time.time_ns()}')
                            if folder.resolve().parent!=OUT.resolve() or target.resolve().parent!=OUT.resolve():raise ValueError('Result path escapes root')
                            folder.rename(target)
                        runner=BenchmarkRunner(c,folder,mass_schedule=binding['mass_schedule'],mass_source_binding=binding['source'])
                        dump(folder/'diagnostic-inputs.json',binding)
                        data=DATA/f'femnist-{c.seed}'
                        x,y,ex,ey=[np.load(data/f'{name}.npy',allow_pickle=False) for name in
                                  ('features','labels','development_features','development_labels')]
                        summary=runner.run_arrays(x,y,lambda:create_model(c.dataset,x.shape[1:],c.num_classes,c.model),(ex,ey))
                        if inputs()!=frozen:raise ValueError('Inputs changed during cell')
                        seal(folder)
                    record=dict(summary,arm=row['arm'],budget=row['budget'])
                    record['trace_metrics']=trace_metrics(folder,summary)
                    records.append(record);state['completed']+=1
                except Exception:
                    state['failed']+=1
                    with (JOB/'failures.jsonl').open('a',encoding='utf-8') as f:
                        f.write(json.dumps({'experiment_id':c.experiment_id,'error':traceback.format_exc()})+'\n')
                analyze(records,state);dump(JOB/'status.json',state)
            state['status']='completed_with_gaps' if state['failed'] or state['held'] else 'completed'
        except BaseException:
            state.update(status='failed',error=traceback.format_exc());raise
        finally:
            state['finished_utc']=datetime.now(timezone.utc).isoformat()
            dump(JOB/'status.json',state);analyze(records,state)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--freeze',action='store_true');group.add_argument('--run',action='store_true')
    parser.add_argument('--wait-for-v11',action='store_true')
    args=parser.parse_args()
    if args.run:run(args.wait_for_v11)
    elif args.freeze:
        freeze();print(f'Frozen {len(manifest())} V12 clean cells; training not started.')
    else:print(json.dumps(manifest(),indent=2))
