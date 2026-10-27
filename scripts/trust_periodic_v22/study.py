"""36 matched online development cells, clean qualification before attack runs."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_name,'2')
import argparse
from dataclasses import asdict,replace
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from scripts.trust_anchor_v20 import study as v20
from scripts.trust_supervised_v18 import study as parent
from scripts.trust_alignment_v17 import study as v17
from scripts.trust_repair_v16_fixed import study as v16
from scripts.trust_repair_v16_fixed.integrity import snapshot,state_hash,assert_hash
from scripts.trust_repair_v16_fixed.research.models import create_model
from scripts.trust_repair_v16_fixed.research.splits import make_splits
from scripts.trust_repair_v16_fixed.runtime import exclusive_lock
from .runner import BenchmarkRunner
from .build_sources import runner_source as rendered_runner
from .online import OnlineRepair,DEPLOYMENT,MemoryPolicy
from .analysis import SEEDS,ATTACKS,ARMS,clean_gate,qualify,extract
from scripts.trust_online_v21_fixed import study as initial
from scripts.trust_online_v21_fixed.parity import historical_weights

ROOT=parent.ROOT
JOB=ROOT/'artifacts/periodic-v22'
OUT=ROOT/'results/periodic-v22'
POLICY_FILE=ROOT/'docs/research/PERIODIC_V22.md'
TEST_FILE=ROOT/'tests/test_periodic_v22.py'
LAUNCHER=ROOT/'scripts/run_periodic_v22.ps1'

read,dump,digest=parent.read,parent.dump,parent.digest


def config(seed,attack,arm):
    if seed not in SEEDS or attack not in ATTACKS or arm not in ARMS:raise ValueError('unknown online cell')
    base=v17.config('femnist','clip_only' if arm=='clip_only' else 'current',seed,attack)
    return replace(base,experiment_id=f'v22-femnist-{arm}-{attack}-s{seed}')


def manifest():
    return [dict(seed=s,attack=t,arm=a,id=config(s,t,a).experiment_id)
            for t in ATTACKS for s in SEEDS for a in ARMS]


def source_folder(seed,attack,arm):
    c=v16.config('femnist','clip_only' if arm=='clip_only' else 'current',seed,attack)
    return v16.OUT/c.experiment_id


def inputs():
    if json.loads(json.dumps(initial.inputs()))!=read(initial.JOB/'frozen-inputs.json'):
        raise ValueError('original frozen V21 source changed')
    v20.assert_frozen()
    if read(v20.JOB/'status.json')['status']!='checkpoint_promising' or not read(v20.JOB/'audit.json')['passed']:
        raise ValueError('V20 did not qualify with completed audit')
    if (Path(__file__).parent/'runner.py').read_text(encoding='utf-8')!=rendered_runner():
        raise ValueError('online runner differs from explicit inherited insertions')
    refs={}
    for t in ATTACKS:
        for s in SEEDS:
            for a in ('clip_only','current'):
                path=source_folder(s,t,a)
                refs[str(path.relative_to(ROOT))]={name:digest(path/name) for name in
                    ('completion.json','summary.json','chain-trace.json')}
    return dict(source={str(p.relative_to(ROOT)):digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        initial_freeze_sha256=digest(initial.JOB/'frozen-inputs.json'),
        failure_audit_sha256=digest(ROOT/'docs/research/ONLINE_V21_FAILURE_AUDIT.md'),
        parent_freeze_sha256=digest(v20.JOB/'frozen-inputs.json'),v20_analysis_sha256=digest(v20.JOB/'analysis.json'),
        v20_audit_sha256=digest(v20.JOB/'audit.json'),historical_references=refs,
        memory_policy=asdict(MemoryPolicy()),deployment=asdict(DEPLOYMENT),repair=asdict(v20.POLICY),anchor=asdict(v20.ANCHOR),
        manifest=[dict(r,config=config(r['seed'],r['attack'],r['arm']).to_dict()) for r in manifest()],
        policy_sha256=digest(POLICY_FILE),tests_sha256=digest(TEST_FILE),launcher_sha256=digest(LAUNCHER),
        clean_first=True,automatic_followup=False,publication_ready=False)


def assert_frozen():
    value=json.loads(json.dumps(inputs()))
    if value!=read(JOB/'frozen-inputs.json'):raise ValueError('frozen V21 inputs changed')


def trace_hashes(folder):
    return [r['model_hash'] for r in read(folder/'chain-trace.json')]


def validate_parity(folder,reference,arm):
    left=read(folder/'summary.json');right=read(reference/'summary.json')
    ignored={'experiment_id','protocol_version'}
    if {k:v for k,v in left['configuration'].items() if k not in ignored}!={k:v for k,v in right['configuration'].items() if k not in ignored}:
        raise ValueError('historical config differs')
    for key in ('training_binding','initial_model_hash','data_fingerprint','test_fingerprint',
                'participants','split_bindings','local_training_work'):
        if left[key]!=right[key]:raise ValueError(f'historical matching mismatch: {key}')
    a,b=trace_hashes(folder),trace_hashes(reference)
    expected=101 if arm in ('clip_only','current') else DEPLOYMENT.first_round
    if len(a)!=101 or a[:expected]!=b[:expected]:raise ValueError('historical model hash parity failed')
    if arm in ('clip_only','current'):
        for key in ('accuracy_history','targeted_attack_success_history','client_rejection_counts',
                    'validation_work'):
            if left[key]!=right[key]:raise ValueError(f'baseline outcome parity failed: {key}')
    reconstructed=historical_weights(reference,right)
    if expected==101 and left['accepted_malicious_weight_history']!=reconstructed:
        raise ValueError('baseline reconstructed malicious weight mismatch')
    return dict(passed=True,checked_models=expected,full_trajectory=expected==101,
                historical_weight_source='accepted IDs and local sample-count events',
                reference_completion_sha256=digest(reference/'completion.json'))


class ProgressLogger:
    """Passive round progress for background monitoring; no feedback to training."""
    def __init__(self,logger,folder,cell_id):
        self.logger,self.folder,self.cell_id=logger,folder,cell_id
        self.attempts=self.admitted=0

    def log(self,event_type,round_index,payload):
        self.logger.log(event_type,round_index,payload)
        if event_type=='online_repair':
            self.attempts+=1;self.admitted+=int(payload['decision']['passed'])
        if event_type=='round_completed':
            dump(self.folder/'progress.json',dict(cell=self.cell_id,round=round_index,total_rounds=100,
                accuracy=payload['accuracy'],repair_attempts=self.attempts,repair_admissions=self.admitted,
                repair_holds=self.attempts-self.admitted,runtime_seconds=payload['runtime_seconds'],
                updated_utc=datetime.now(timezone.utc).isoformat()))
            if round_index==1 or round_index%10==0:
                print(f"ROUND {round_index}/100 {self.cell_id}; repairs {self.admitted}/{self.attempts}",flush=True)


class OfflineObserver:
    """Evaluation only after admission. No return path into the controller."""
    def __init__(self,features,labels,target,clients):
        self.features,self.labels,self.target,self.clients=features,labels,target,clients
        self.records=[]

    def __call__(self,outcome,rnd):
        started=time.perf_counter()
        source=parent.evaluate(outcome.source,self.features,self.labels,self.target,self.clients)
        raw=parent.evaluate(outcome.candidate,self.features,self.labels,self.target,self.clients)
        committed=raw if outcome.payload['decision']['passed'] else source
        assert_hash(outcome.committed,committed['model_hash'],'offline committed equality')
        self.records.append(dict(round=rnd,source=source,candidate=raw,committed=committed,
                                 admitted=outcome.payload['decision']['passed'],seconds=time.perf_counter()-started))


def collect_rows():
    rows=[]
    for row in manifest():
        folder=OUT/row['id']
        if (folder/'completion.json').exists():
            parent.verify_seal(folder);rows.append(read(folder/'online-summary.json'))
    return rows


def require_clean_gate():
    gate=clean_gate(collect_rows())
    if not gate.get('passed'):raise ValueError('clean qualification required before attacked cells')
    return gate


def execute(index):
    with exclusive_lock(JOB/'cell.lock'),threadpool_limits(limits=2):
        assert_frozen();row=manifest()[index];c=config(row['seed'],row['attack'],row['arm']);folder=OUT/row['id']
        if row['attack']!='none':require_clean_gate()
        if folder.exists():raise FileExistsError('refusing existing online cell')
        torch.set_num_threads(2)
        if not torch.cuda.is_available():raise RuntimeError('CUDA required for historical parity')
        reference=source_folder(row['seed'],row['attack'],row['arm'])
        if read(reference/'completion.json')!=v16.sealed_files(reference):raise ValueError('historical completion seal mismatch')
        x,y,tx,ty=parent.load_arrays(row['seed']);split=make_splits(y,c,external_test=True)
        runner=BenchmarkRunner(c,folder)
        runner.logger=ProgressLogger(runner.logger,folder,row['id'])
        runner.challenge_arm='clip_only' if row['arm']=='clip_only' else 'current'
        observer=OfflineObserver(tx,ty,c.backdoor_target_label,split.natural_test_clients)
        if row['arm'] in ('v20_periodic','v20_peak'):
            runner.online_controller=OnlineRepair(row['arm'],DEPLOYMENT,v20.POLICY,v20.ANCHOR)
            runner.offline_observer=observer
        runner.run_arrays(x,y,lambda:create_model(c.dataset,x.shape[1:],c.num_classes,c.model),(tx,ty))
        parity=validate_parity(folder,reference,row['arm']);dump(folder/'parity.json',parity)
        final=snapshot(runner.final_model);final_hash=state_hash(final)
        if final_hash!=trace_hashes(folder)[-1]:raise ValueError('final state differs from trace')
        torch.save(dict(model_hash=final_hash,state=final),folder/'final.pt')
        events=[json.loads(line) for line in (folder/'events.jsonl').open(encoding='utf-8')]
        integrity=[e for e in events if e['event_type']=='model_integrity']
        if [e['round_index'] for e in integrity]!=list(range(1,101)) or not all(
                e['payload']['passed'] and e['payload']['committed_hash']==e['payload']['expected_hash'] for e in integrity):
            raise ValueError('missing global integrity checks')
        online=[e['payload'] for e in events if e['event_type']=='online_repair']
        monitors=[e['payload'] for e in events if e['event_type']=='private_reference_monitor']
        if row['arm'] in ('v20_periodic','v20_peak') and [m['round'] for m in monitors]!=list(range(1,101)):
            raise ValueError('incomplete private running-peak monitoring')
        expected=[25,50,75,100] if row['arm'] in ('v20_periodic','v20_peak') else []
        if [r['round'] for r in online]!=expected or [r['round'] for r in observer.records]!=expected:
            raise ValueError('online schedule/evaluation mismatch')
        for r in online:
            if r['committed_model_hash']!=trace_hashes(folder)[r['round']]:raise ValueError('online commit trace mismatch')
        dump(folder/'offline-repair-evaluation.json',dict(records=observer.records))
        audit=dict(passed=True,arm=row['arm'],checked_rounds=100,online_decisions=len(online),
                   final_model_hash=final_hash,parity=parity)
        dump(folder/'integrity-summary.json',audit)
        summary=extract(read(folder/'summary.json'),online,observer.records,audit,events)
        final_eval=parent.evaluate(runner.final_model,tx,ty,c.backdoor_target_label,split.natural_test_clients)
        if final_eval['model_hash']!=final_hash or final_eval['accuracy']!=summary['final_accuracy']:
            raise ValueError('final offline evaluation mismatch')
        dump(folder/'final-evaluation.json',final_eval)
        summary['final_evaluation']=final_eval
        summary['private_reference_presentations']=sum(m['example_presentations'] for m in monitors)
        summary['private_reference_transmitted_bytes']=sum(m['extra_transmitted_bytes'] for m in monitors)
        summary['memory_candidate_presentations']=sum(r['memory_candidate_presentations'] for r in online)
        summary['extra_repair_example_presentations']+=summary['private_reference_presentations']+summary['memory_candidate_presentations']
        summary['extra_repair_transmitted_bytes']+=summary['private_reference_transmitted_bytes']
        summary['private_reference_seconds']=sum(m['seconds'] for m in monitors)
        if row['attack']=='model_replacement':
            history=read(folder/'summary.json')['targeted_attack_success_history']
            summary['reinfection']=[dict(round=r['round'],pre_repair_trigger_rate=r['source']['trigger_target_rate'],
                previous_committed_trigger_rate=history[r['round']-2],
                increase=r['source']['trigger_target_rate']-history[r['round']-2]) for r in observer.records]
        dump(folder/'online-summary.json',summary)
        dump(folder/'settings.json',dict(row=row,freeze_sha256=digest(JOB/'frozen-inputs.json')))
        assert_frozen();dump(folder/'completion.json',parent.seal_files(folder));parent.verify_seal(folder)


def report():
    rows=collect_rows();verdict=qualify(rows)
    dump(JOB/'analysis.json',verdict);dump(JOB/'clean-gate.json',clean_gate(rows))
    dump(JOB/'results.json',dict(records=rows,analysis=verdict))
    lines=['# V22 periodic repair and private memory study','',f"Status: {verdict['status']}; cells completed: {len(rows)}/36.",
        'Fixed repair rounds: 25, 50, 75, 100. Private fixed-example peaks are monitored every round. Veto retains the quorum aggregate.',
        'All clean cells precede attacks; attack runs require the frozen clean qualification.','',
        '| Attack | Seed | Arm | Final accuracy | Mean post-onset ASR | Endpoint ASR | Repairs admitted/attempted | Lower-decile utility |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    for r in rows:
        rate=lambda key:'n/a' if r[key] is None else f'{100*r[key]:.2f}%'
        lines.append(f"| {r['attack']} | {r['seed']} | {r['arm']} | {100*r['final_accuracy']:.2f}% | {rate('mean_post_onset_asr')} | {rate('endpoint_asr')} | {r['repair_admissions']}/{r['repair_attempts']} | {100*r['lower_decile_client_utility']:.2f}% |")
    lines+=['','Full per-round pre/post repair metrics, conditional ASR denominators, holds, honest exclusions,',
        'accepted malicious aggregation weight, verification cost, teacher work and simulated communication',
        'are in the cell JSON artifacts. Repeated validation is empirical; role/example reuse is disclosed.',
        'These are exposed-seed development runs. No automatic confirmation or acceptance-readiness claim.']
    (JOB/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return verdict


def smoke():
    from scripts.trust_online_v21 import study as original
    if not read(original.JOB/'smoke-validation.json')['passed']:
        raise ValueError('missing unchanged-inner-loop GPU validation')
    dump(JOB/'smoke-validation.json',dict(passed=True,inner_loop_unchanged=True,
        source_sha256=digest(original.JOB/'smoke-validation.json'),original_gpu_steps=400))


def run():
    with exclusive_lock(JOB/'coordinator.lock'):
        assert_frozen()
        previous=read(initial.JOB/'status.json')
        if previous['completed']!=12 or previous['failed'] or previous['status']!='clean_rejected':
            raise ValueError('V21-fixed must complete and reject its clean screen before V22 starts')
        if not read(JOB/'validation.json')['passed'] or not read(JOB/'smoke-validation.json')['passed'] or not read(JOB/'allocation-validation.json')['passed']:
            raise ValueError('tests and real GPU smoke must pass before launch')
        state=dict(status='running',pid=os.getpid(),completed=0,total=36,failed=0,held=0,
                   started_utc=datetime.now(timezone.utc).isoformat(),publication_ready=False)
        dump(JOB/'status.json',state)
        dump(JOB/'environment.json',dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0)))
        try:
            for i,row in enumerate(manifest()):
                assert_frozen()
                if i>=12:
                    gate=clean_gate(collect_rows())
                    if not gate.get('passed'):
                        state.update(status='clean_rejected',held=24,finished_utc=datetime.now(timezone.utc).isoformat())
                        report();dump(JOB/'status.json',state);return
                state.update(current=row['id'],phase='clean' if i<12 else 'attacked',
                             updated_utc=datetime.now(timezone.utc).isoformat())
                dump(JOB/'status.json',state);folder=OUT/row['id']
                if (folder/'completion.json').exists():
                    parent.verify_seal(folder)
                    if read(folder/'settings.json')!=dict(row=row,freeze_sha256=digest(JOB/'frozen-inputs.json')):
                        raise ValueError('completed cell settings mismatch')
                else:
                    if folder.exists():
                        if not folder.resolve().is_relative_to(OUT.resolve()):raise ValueError('unsafe partial output location')
                        folder.rename(folder.with_name(folder.name+f'.interrupted-{time.time_ns()}'))
                    print(f"START {i+1}/36 {row['id']}",flush=True)
                    result=subprocess.run([sys.executable,'-u','-m','scripts.trust_periodic_v22.study','--cell',str(i)],cwd=ROOT)
                    if result.returncode:raise RuntimeError(f"worker failed {row['id']}: {result.returncode}")
                    parent.verify_seal(folder)
                state['completed']+=1;verdict=report();dump(JOB/'status.json',state)
                print(f"DONE {i+1}/36 {row['id']}",flush=True)
            state.update(status=verdict['status'],finished_utc=datetime.now(timezone.utc).isoformat())
            dump(JOB/'status.json',state)
        except Exception as error:
            state.update(status='failed',failed=1,error=str(error),traceback=traceback.format_exc())
            dump(JOB/'status.json',state);raise


def main():
    parser=argparse.ArgumentParser();action=parser.add_mutually_exclusive_group(required=True)
    for option in ('freeze','run','report','smoke'):action.add_argument('--'+option,action='store_true')
    action.add_argument('--cell',type=int);args=parser.parse_args()
    if args.freeze:
        value=json.loads(json.dumps(inputs()));path=JOB/'frozen-inputs.json'
        if path.exists() and read(path)!=value:raise ValueError('V21 already frozen differently')
        dump(path,value);print(f'Frozen V22: {digest(path)}')
    elif args.run:run()
    elif args.report:assert_frozen();report()
    elif args.smoke:smoke()
    else:execute(args.cell)


if __name__=='__main__':main()
