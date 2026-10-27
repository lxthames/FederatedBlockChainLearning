"""Finite clean-first security pilot for the frozen wide-CNN candidate."""
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
JOB=ROOT/'artifacts/security-v13'
OUT=ROOT/'results/security-v13'
PARENT=ROOT/'artifacts/budget-v12'
POLICY=ROOT/'docs/research/SECURITY_V13.md'
AUDIT=ROOT/'docs/research/TRUST_EXTENSION_BASELINE_AUDIT.md'
SEEDS=(11,17,23)
DATASETS=('fashion_mnist','femnist')
CONTROLS=('fedavg','clip_only','fedval')
CANDIDATE='scaled_renormalized'
ARMS=CONTROLS+(CANDIDATE,)
ATTACKS=('low_and_slow','model_replacement')
FILES=('config.json','summary.json','splits.json','events.jsonl','chain-trace.json','diagnostic-inputs.json')
EMPTY_BINDING={'mass_schedule':None,'source':None}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def config(dataset,arm,seed,attack='none'):
    if dataset not in DATASETS or arm not in ARMS or seed not in SEEDS or attack not in ('none',)+ATTACKS:
        raise ValueError('Unknown V13 cell')
    femnist=dataset=='femnist'
    return ExperimentConfig(experiment_id=f'v13-{dataset}-{arm}-{attack}-s{seed}',
        dataset=dataset,num_clients=50,participation_rate=.4,malicious_fraction=.3,
        seed=seed,rounds=40 if femnist else 30,attack=attack,attack_start_round=11,
        num_classes=62 if femnist else 10,batch_size=16 if femnist else 64,
        learning_rate=.05 if femnist else .01,optimizer_momentum=.9,device='cuda',
        dirichlet_alpha=None if femnist else .5,sampling_policy='bounded',model='wide_cnn',
        local_epochs=1,local_training_budget='epoch',aggregation_mass_policy='renormalized',
        defense=arm if arm in ('fedavg','fedval') else 'two_layer',
        vote_rule='scaled_loss' if arm==CANDIDATE else 'rank_loss',ablation='validation_only',
        candidate_clip_ratio=2.,natural_splits_path=str(DATA/f'{dataset}-{seed}/natural-splits.json') if femnist else None,
        validation_distribution='private',trust_updates='fixed',trust_weighting='fixed',trust_eligibility='fixed',
        trust_reward=0.,rejection_trust_penalty=0.,rejection_trust_policy='none',
        ranking_keep_fraction=.8 if arm==CANDIDATE else 1.,loss_tolerance=0.,loss_standard_errors=2.)


def manifest():
    # Fashion clean check first; every clean cell precedes every attack cell.
    conditions=('none',)+ATTACKS
    return [{'dataset':d,'arm':a,'seed':s,'attack':t,'config':config(d,a,s,t).to_dict()}
            for t in conditions for d in DATASETS for s in SEEDS for a in ARMS]


def inputs():
    data={}
    for dataset in DATASETS:
        for seed in SEEDS:
            folder=DATA/f'{dataset}-{seed}';provenance=read(folder/'provenance.json')
            names=['features.npy','labels.npy','development_features.npy','development_labels.npy']
            if dataset=='femnist':names.append('natural-splits.json')
            for name in names:
                actual=digest(folder/name)
                if provenance['files'].get(name)!=actual:raise ValueError('Prepared data changed')
                data[str((folder/name).relative_to(ROOT))]=actual
            data[str((folder/'provenance.json').relative_to(ROOT))]=digest(folder/'provenance.json')
    return {'source':{str(p.relative_to(ROOT)):digest(p) for p in sorted(Path(__file__).parent.rglob('*.py'))},
        'data':data,'manifest':manifest(),'policy_sha256':digest(POLICY),'baseline_audit_sha256':digest(AUDIT),
        'parent_freeze_sha256':digest(PARENT/'frozen-inputs.json'),
        'evaluation_role':'search-exposed clean-first attack development pilot',
        'automatic_confirmation':False,'publication_ready':False}


def freeze():
    value=inputs();path=JOB/'frozen-inputs.json'
    if path.exists() and read(path)!=value:raise ValueError('Frozen V13 changed')
    dump(path,value)
    return value


def verify_parent():
    state=read(PARENT/'status.json')
    if state['status']!='completed' or state['completed']!=60 or state['failed'] or state['held']:
        raise RuntimeError('V12 must complete successfully before V13')
    with exclusive_lock(PARENT/'coordinator.lock'):pass
    frozen=read(PARENT/'frozen-inputs.json')
    for relative,expected in {**frozen['source'],**frozen['data']}.items():
        if digest(ROOT/relative)!=expected:raise ValueError('Parent source/data changed')
    if digest(ROOT/'docs/research/BUDGET_V12.md')!=frozen['policy_sha256']:
        raise ValueError('Parent protocol changed')
    for row in frozen['manifest']:
        folder=ROOT/'results/budget-v12'/row['config']['experiment_id']
        if read(folder/'completion.json')!={name:digest(folder/name) for name in FILES}:
            raise ValueError('Parent completion seal changed')
        expected=hashlib.sha256(json.dumps(row['config'],sort_keys=True).encode()).hexdigest()
        summary=read(folder/'summary.json')
        if summary.get('config_hash')!=expected or summary.get('status')!='completed':
            raise ValueError('Parent completed configuration changed')
    return state


def seal(folder):dump(folder/'completion.json',{name:digest(folder/name) for name in FILES})


def load_completed(folder,c):
    if read(folder/'completion.json')!={name:digest(folder/name) for name in FILES}:
        raise ValueError('Completed artifact digest mismatch')
    summary=read(folder/'summary.json')
    expected=hashlib.sha256(json.dumps(c.to_dict(),sort_keys=True).encode()).hexdigest()
    if summary.get('config_hash')!=expected or summary.get('status')!='completed':
        raise ValueError('Completed configuration/status mismatch')
    if read(folder/'diagnostic-inputs.json')!=EMPTY_BINDING or summary.get('diagnostic_input_binding')!=EMPTY_BINDING:
        raise ValueError('Unexpected external aggregation input')
    return summary


def enrich(summary,folder):
    """Derived annotations live in analysis, never in immutable summaries."""
    result=dict(summary);c=summary['configuration'];onset=c['attack_start_round']
    post=summary['targeted_attack_success_history'][onset-1:]
    clean=summary.get('clean_trigger_prediction_history',[])[onset-1:]
    result['post_onset_mean_asr']=float(np.mean(post)) if c['attack']=='model_replacement' and post else None
    result['endpoint_asr']=float(post[-1]) if c['attack']=='model_replacement' and post else None
    result['post_onset_mean_clean_trigger_rate']=float(np.mean(clean)) if clean else None
    trace=read(Path(folder)/'chain-trace.json')
    before=[r for r in trace if r.get('round',0)<onset]
    if [r.get('round',0) for r in before]!=list(range(min(onset,c['rounds']+1))):
        raise ValueError('Incomplete pre-onset trace')
    result['pre_onset_model_hashes']=[r['model_hash'] for r in before]
    rejection=summary['client_rejection_counts'];submission=summary['client_submission_counts']
    honest=[i for i in range(c['num_clients']) if c['attack']=='none' or i not in summary['malicious_client_ids']]
    get=lambda counts,i:counts.get(str(i),counts.get(i,0))
    result['honest_update_exclusion_rate']=sum(get(rejection,i) for i in honest)/max(1,sum(get(submission,i) for i in honest))
    return result


def training_binding_valid(row):
    try:return row['training_binding']==digest_value({k:row['configuration'][k] for k in TRAINING_FIELDS})
    except KeyError:return False


def method_pair(left,right):
    pair=compare_pair(left,right)
    if not training_binding_valid(left) or not training_binding_valid(right):
        pair={'matched':False,'mismatches':pair['mismatches']+['invalid_training_binding']}
    ignored={'experiment_id','defense','vote_rule','ranking_keep_fraction'}
    if {k:v for k,v in left.get('configuration',{}).items() if k not in ignored}!={k:v for k,v in right.get('configuration',{}).items() if k not in ignored}:
        pair={'matched':False,'mismatches':pair['mismatches']+['undeclared method configuration change']}
    return pair


def condition_pair(clean,attacked):
    """Only attack type may change; common pre-onset models are required."""
    if clean is None or attacked is None:return {'matched':False,'mismatches':['missing condition']}
    reasons=[]
    if clean['attack']!='none' or attacked['attack'] not in ATTACKS:reasons.append('unexpected condition')
    if not training_binding_valid(clean) or not training_binding_valid(attacked):reasons.append('invalid_training_binding')
    a,b=clean['configuration'],attacked['configuration']
    if any(a.get(k)!=b.get(k) for k in TRAINING_FIELDS if k!='attack'):reasons.append('undeclared training change')
    # Defense configuration must also be identical across its clean/attack runs.
    ignored={'attack','experiment_id'}
    if {k:v for k,v in a.items() if k not in ignored}!={k:v for k,v in b.items() if k not in ignored}:
        reasons.append('undeclared configuration change')
    l,r=dict(clean),dict(attacked)
    l['training_binding']=r['training_binding']='declared attack intervention'
    pair=compare_pair(l,r)
    if not clean.get('pre_onset_model_hashes') or clean.get('pre_onset_model_hashes')!=attacked.get('pre_onset_model_hashes'):
        reasons.append('pre-onset models differ')
    return {'matched':not reasons and pair['matched'],'mismatches':reasons+pair['mismatches']}


def find(records,arm,dataset,seed,attack='none'):
    return next((r for r in records if (r['arm'],r['dataset'],r['seed'],r['attack'])==(arm,dataset,seed,attack)),None)


def clean_cell(records,arm,dataset,seed):
    baseline=find(records,'fedavg',dataset,seed);candidate=find(records,arm,dataset,seed)
    if baseline is None or candidate is None:return {'passed':False,'reason':'missing clean cell','dataset':dataset,'seed':seed}
    pair=method_pair(candidate,baseline);delta=candidate['final_utility']-baseline['final_utility']
    return {'passed':bool(pair['matched'] and baseline['final_utility']>=.2 and delta>=-.05
        and candidate['unsupported_rounds']<=3 and candidate['honest_validator_penalty_amount']==0),
        'accuracy_difference':delta,'pairing':pair,'dataset':dataset,'seed':seed}


def candidate_gate(records):
    cells=[clean_cell(records,CANDIDATE,d,s) for d in DATASETS for s in SEEDS]
    return {'passed':all(r['passed'] for r in cells),'cells':cells,
        'rule':'All six matched clean cells: FedAvg >=20%, candidate loss <=5 points, unsupported <=3, zero honest monetary penalty.'}


def effectiveness(records,dataset,seed,attack):
    clean=find(records,'fedavg',dataset,seed);attacked=find(records,'fedavg',dataset,seed,attack)
    pair=condition_pair(clean,attacked)
    if not pair['matched']:return {'effective':None,'pairing':pair,'reason':'missing or mismatched FedAvg conditions'}
    if attack=='low_and_slow':
        damage=clean['final_utility']-attacked['final_utility']
        return {'effective':damage>=.01,'utility_damage':damage,'pairing':pair}
    asr=attacked['post_onset_mean_asr'];trigger=clean['post_onset_mean_clean_trigger_rate']
    if asr is None or trigger is None:return {'effective':None,'reason':'missing post-onset trigger measurements','pairing':pair}
    return {'effective':asr-trigger>=.10,'asr_increase_over_clean':asr-trigger,'pairing':pair}


def analyze(records,state):
    comparisons=[]
    for row in records:
        d,s,t=row['dataset'],row['seed'],row['attack']
        own_clean=find(records,row['arm'],d,s)
        own_pair=condition_pair(own_clean,row) if t!='none' else {'matched':True,'mismatches':[]}
        for reference in CONTROLS:
            baseline=find(records,reference,d,s,t)
            if baseline is None:continue
            pair=method_pair(row,baseline)
            reference_pair=condition_pair(find(records,reference,d,s),baseline) if t!='none' else {'matched':True,'mismatches':[]}
            valid=pair['matched'] and own_pair['matched'] and reference_pair['matched']
            comparisons.append({'dataset':d,'seed':s,'attack':t,'arm':row['arm'],'reference':reference,
                **pair,'pre_onset_conditions_matched':own_pair['matched'] and reference_pair['matched'],
                'valid':valid,'clean_gate':clean_cell(records,row['arm'],d,s),
                'reference_clean_gate':clean_cell(records,reference,d,s),
                'accuracy_difference':row['final_utility']-baseline['final_utility'] if valid else None,
                'asr_reduction':baseline['post_onset_mean_asr']-row['post_onset_mean_asr'] if valid and t=='model_replacement' else None})
    groups=[]
    for d in DATASETS:
        for arm in ARMS:
            for t in ('none',)+ATTACKS:
                rows=[r for r in records if (r['dataset'],r['arm'],r['attack'])==(d,arm,t)]
                pairs=[r for r in comparisons if (r['dataset'],r['arm'],r['attack'],r['reference'])==(d,arm,t,'fedavg')]
                complete=sorted(r['seed'] for r in rows)==list(SEEDS) and len(pairs)==3 and all(r['valid'] for r in pairs)
                mean=lambda key:float(np.mean([r[key] for r in rows])) if complete else None
                groups.append({'dataset':d,'arm':arm,'attack':t,'complete':complete,'seeds':[r['seed'] for r in rows],
                    'mean_accuracy':mean('final_utility'),
                    'mean_post_onset_asr':mean('post_onset_mean_asr') if t=='model_replacement' else None,
                    'mean_endpoint_asr':mean('endpoint_asr') if t=='model_replacement' else None,
                    'mean_honest_exclusion':mean('honest_update_exclusion_rate'),
                    'all_clean_cells_pass':all(p['clean_gate']['passed'] for p in pairs) if complete else None})
    attack_checks=[{'dataset':d,'seed':s,'attack':t,**effectiveness(records,d,s,t)}
                   for d in DATASETS for s in SEEDS for t in ATTACKS]
    screens=[]
    for d in DATASETS:
        for t in ATTACKS:
            for reference in CONTROLS:
                pairs=[p for p in comparisons if (p['dataset'],p['attack'],p['arm'],p['reference'])==(d,t,CANDIDATE,reference)]
                checks=[r for r in attack_checks if (r['dataset'],r['attack'])==(d,t)]
                complete=sorted(p['seed'] for p in pairs)==list(SEEDS) and all(p['valid'] for p in pairs)
                effective=all(r['effective'] is True for r in checks)
                clean_ok=candidate_gate(records)['passed']
                reference_ok=complete and all(p['reference_clean_gate']['passed'] for p in pairs)
                gain=float(np.mean([p['accuracy_difference'] for p in pairs])) if complete else None
                asr_gain=float(np.mean([p['asr_reduction'] for p in pairs])) if complete and t=='model_replacement' else None
                eligible=complete and effective and clean_ok and reference_ok
                attacked_rows=[find(records,CANDIDATE,d,s,t) for s in SEEDS]
                penalties_ok=all(r is not None and r['honest_validator_penalty_amount']==0 for r in attacked_rows)
                passes=bool(eligible and penalties_ok and all(p['accuracy_difference']>=-.05 for p in pairs)
                    and (gain>=.01 if t=='low_and_slow' else asr_gain>=.05))
                screens.append({'dataset':d,'attack':t,'reference':reference,'complete':complete,
                    'all_attacks_effective':effective,'candidate_clean_gate':clean_ok,'reference_clean_gate':bool(reference_ok),
                    'mean_accuracy_gain':gain,'mean_asr_reduction':asr_gain,
                    'interpretable':bool(eligible),'descriptive_benefit_screen_passed':passes if eligible else None})
    compact=[{k:r[k] for k in ('experiment_id','dataset','seed','attack','arm','final_utility','post_onset_mean_asr',
        'endpoint_asr','post_onset_mean_clean_trigger_rate','lower_decile_client_utility','honest_update_exclusion_rate',
        'malicious_update_acceptance_rate','unsupported_rounds','local_training_work','validation_work','runtime_seconds_total')}
        for r in records]
    dump(JOB/'analysis.json',{'status':state,'runs':compact,'groups':groups,'comparisons':comparisons,
        'attack_effectiveness':attack_checks,'candidate_gate':candidate_gate(records),'benefit_screens':screens,
        'automatic_promotion':False,'publication_ready':False})
    lines=['# V13 wide-CNN clean-first security pilot','',
        '| Dataset | Method | Condition | Mean accuracy | Mean post-onset ASR |','|---|---|---|---:|---:|']
    for g in groups:
        accuracy=f"{100*g['mean_accuracy']:.2f}%" if g['complete'] else 'incomplete / held'
        asr=f"{100*g['mean_post_onset_asr']:.2f}%" if g['mean_post_onset_asr'] is not None else 'n/a'
        lines.append(f"| {g['dataset']} | {g['arm']} | {g['attack']} | {accuracy} | {asr} |")
    lines+=['',f"Candidate clean gate: {candidate_gate(records)['passed']}.",
        'Means require all three matched seeds and verified pre-onset traces. Conditions with ineffective attacks do not establish defense benefit.',
        'FedVal is an audited paper-based adapter with different central-reference access and validation work.',
        'Exposed development seeds; no independent confirmation, significance or novelty claim. No automatic confirmation launch.','']
    (JOB/'comparison.md').write_text('\n'.join(lines),encoding='utf-8')


def run():
    with exclusive_lock(JOB/'coordinator.lock'),threadpool_limits(limits=2):
        frozen=read(JOB/'frozen-inputs.json')
        if inputs()!=frozen:raise ValueError('Frozen V13 changed')
        parent=verify_parent()
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        torch.set_num_threads(2)
        dump(JOB/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,
            'numpy':np.__version__,'gpu':torch.cuda.get_device_name(0),'parent_final_status':parent})
        state={'status':'running','pid':os.getpid(),'completed':0,'held':0,'failed':0,'total':len(frozen['manifest']),
               'publication_ready':False,'started_utc':datetime.now(timezone.utc).isoformat()}
        records=[];dump(JOB/'status.json',state)
        try:
            for row in frozen['manifest']:
                if inputs()!=frozen:raise ValueError('Frozen inputs changed during V13')
                c=ExperimentConfig(**row['config']);folder=OUT/c.experiment_id
                state['current']=c.experiment_id;dump(JOB/'status.json',state)
                if c.attack!='none' and row['arm']==CANDIDATE and not candidate_gate(records)['passed']:
                    dump(folder/'held.json',candidate_gate(records));state['held']+=1
                    analyze(records,state);dump(JOB/'status.json',state);continue
                try:
                    if (folder/'completion.json').exists():summary=load_completed(folder,c)
                    else:
                        if folder.exists():
                            target=folder.with_name(folder.name+f'.interrupted-{time.time_ns()}')
                            if folder.resolve().parent!=OUT.resolve() or target.resolve().parent!=OUT.resolve():raise ValueError('Result path escapes root')
                            folder.rename(target)
                        runner=BenchmarkRunner(c,folder);dump(folder/'diagnostic-inputs.json',EMPTY_BINDING)
                        data=DATA/f'{c.dataset}-{c.seed}'
                        x,y,ex,ey=[np.load(data/f'{name}.npy',allow_pickle=False) for name in
                                  ('features','labels','development_features','development_labels')]
                        summary=runner.run_arrays(x,y,lambda:create_model(c.dataset,x.shape[1:],c.num_classes,c.model),(ex,ey))
                        if inputs()!=frozen:raise ValueError('Inputs changed during cell')
                        seal(folder)
                    records.append(dict(enrich(summary,folder),arm=row['arm']));state['completed']+=1
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
    args=parser.parse_args()
    if args.run:run()
    elif args.freeze:
        freeze();print(f'Frozen {len(manifest())} maximum V13 cells; training not started.')
    else:print(json.dumps(manifest(),indent=2))
