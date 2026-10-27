"""Frozen V20 diagnostic. No automatic search or end-to-end promotion."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_name,'2')
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch
from threadpoolctl import threadpool_limits
from scripts.trust_supervised_v18 import study as parent
from scripts.trust_supervised_v18.repair import allocate,partition_binding,verify
from scripts.trust_interleave_v19_fixed import study as previous
from scripts.trust_interleave_v19.repair import make_plan,plan_hash
from scripts.trust_repair_v16_fixed.integrity import snapshot,assert_hash
from scripts.trust_repair_v16_fixed.research.models import create_model
from scripts.trust_repair_v16_fixed.research.splits import make_splits
from scripts.trust_repair_v16_fixed.runtime import exclusive_lock
from .repair import ARMS,AnchorPolicy,repair_model

ROOT=parent.ROOT
JOB=ROOT/'artifacts/anchor-v20'
OUT=ROOT/'results/anchor-v20'
POLICY_FILE=ROOT/'docs/research/ANCHOR_V20.md'
TEST_FILE=ROOT/'tests/test_anchor_v20.py'
LAUNCHER=ROOT/'scripts/run_anchor_v20.ps1'
SEEDS=parent.SEEDS
CONDITIONS=parent.CONDITIONS
POLICY=parent.POLICY
ANCHOR=AnchorPolicy()
read,dump,digest=parent.read,parent.dump,parent.digest


def manifest():
    return [dict(condition=t,seed=s,checkpoint=100,id=f'v20-femnist-{t}-s{s}-r100')
            for t in CONDITIONS for s in SEEDS]


def reference_folder(seed,condition):
    return previous.OUT/f'v19fixed-femnist-interleaved-{condition}-s{seed}-r100'


def inputs():
    previous.assert_frozen()
    state=read(previous.JOB/'status.json')
    if state['completed']!=12 or state['failed']:
        raise ValueError('V19-fixed incomplete or failed')
    refs={}
    for row in manifest():
        folder=reference_folder(row['seed'],row['condition'])
        parent.verify_seal(folder)
        refs[str(folder.relative_to(ROOT))]=digest(folder/'completion.json')
    return dict(source={str(p.relative_to(ROOT)):digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        previous_freeze_sha256=digest(previous.JOB/'frozen-inputs.json'),references=refs,
        manifest=manifest(),arms=ARMS,policy=asdict(POLICY),anchor_policy=asdict(ANCHOR),
        policy_sha256=digest(POLICY_FILE),test_sha256=digest(TEST_FILE),launcher_sha256=digest(LAUNCHER),
        automatic_followup=False,publication_ready=False)


def assert_frozen():
    # Serialize tuples exactly as the on-disk JSON representation.
    import json
    if json.loads(json.dumps(inputs()))!=read(JOB/'frozen-inputs.json'):
        raise ValueError('frozen V20 inputs changed')


def baseline_parity(telemetry,record,reference):
    old=read(reference/'hard_patch-repair.json')
    prior=read(reference/'hard_patch-result.json')
    if telemetry!=old:
        raise ValueError('V19 baseline training telemetry mismatch')
    for key in ('raw','committed','decision','held','parent_model_hash','work','plan_sha256'):
        if record[key]!=prior[key]:
            raise ValueError(f'V19 baseline mismatch: {key}')
    return dict(passed=True,checked_optimizer_steps=len(old['steps']),
                reference_completion_sha256=digest(reference/'completion.json'))


def execute(index):
    with exclusive_lock(JOB/'cell.lock'),threadpool_limits(limits=2):
        assert_frozen();row=manifest()[index];folder=OUT/row['id']
        if folder.exists():
            raise FileExistsError('refusing existing output folder')
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required for historical parity')
        seed,condition=row['seed'],row['condition'];c=parent.replay_config(seed,condition)
        reference=reference_folder(seed,condition)
        v18=previous.reference_folder(seed,condition);replay=parent.OUT/c.experiment_id
        for source in (reference,v18,replay):parent.verify_seal(source)
        x,y,tx,ty=parent.load_arrays(seed);split=make_splits(y,c,external_test=True)
        groups=allocate(split.validation,read(replay/'summary.json')['participants'][99],seed,100,POLICY)
        if partition_binding(groups)!=read(reference/'partitions.json')['binding']:
            raise ValueError('repair/verification split mismatch')
        factory=lambda:create_model(c.dataset,x.shape[1:],c.num_classes,c.model).to('cuda')
        model,checkpoint=parent.load_checkpoint(replay/'checkpoints/round-100.pt',factory)
        local_data={client:(x[ids].copy(),y[ids].copy()) for client,ids in groups['repair'].items()}
        plan=make_plan(local_data,seed,100,POLICY)
        if plan_hash(plan)!=read(reference/'hard_patch-result.json')['plan_sha256']:
            raise ValueError('V19 minibatch/proposal plan mismatch')
        before=previous.restore_panel_order(read(v18/'parent-verification.json'),groups)
        if any(m['model_hash']!=checkpoint['model_hash'] for members in before.values() for m in members.values()):
            raise ValueError('parent measurements do not match checkpoint')
        baseline=read(v18/'unchanged-result.json')['raw']
        folder.mkdir(parents=True);dump(folder/'plan.json',dict(updates=plan,sha256=plan_hash(plan)))
        dump(folder/'partitions.json',read(reference/'partitions.json'))
        records=[]
        for arm in ARMS:
            print(f"{row['id']}: {arm}",flush=True);started=time.perf_counter()
            candidate,telemetry=repair_model(model,local_data,arm,seed,100,POLICY,plan,ANCHOR)
            measured=parent.panel_measurements(candidate,groups,x,y)
            verdict=verify(before,measured,POLICY)
            committed=parent.select_committed(model,candidate,verdict)
            # Evaluation only after a final admission decision, never optimization.
            raw=parent.evaluate(candidate,tx,ty,c.backdoor_target_label,split.natural_test_clients)
            score=raw if verdict['passed'] else baseline
            assert_hash(committed,score['model_hash'],'scored and committed equality')
            record=dict(seed=seed,condition=condition,checkpoint=100,arm=arm,
                parent_model_hash=checkpoint['model_hash'],decision=verdict,raw=raw,committed=score,
                baseline=baseline,held=not verdict['passed'],work=telemetry['work'],
                seconds=time.perf_counter()-started,plan_sha256=telemetry['plan_sha256'])
            record['parity']=baseline_parity(telemetry,record,reference) if arm=='hard_patch' else dict(applicable=False)
            dump(folder/f'{arm}-verification.json',measured);dump(folder/f'{arm}-repair.json',telemetry)
            dump(folder/f'{arm}-result.json',record)
            torch.save(dict(model_hash=telemetry['model_hash'],state=snapshot(candidate)),folder/f'{arm}-candidate.pt')
            torch.save(dict(model_hash=score['model_hash'],state=snapshot(committed)),folder/f'{arm}-committed.pt')
            records.append(record)
        for key in ('optimizer_steps','training_examples','proposal_examples','diagnostic_examples'):
            if len({r['work'][key] for r in records})!=1:
                raise ValueError('student/data budget mismatch')
        assert_hash(model,checkpoint['model_hash'],'parent preserved')
        dump(folder/'summary.json',dict(row=row,records=records,
            verification_work=dict(read(v18/'summary.json')['verification_work'],evaluated_models=4,reused_parent_measurements=1),
            note='Student presentation/update budgets matched; teacher compute and transmissions are additional.'))
        dump(folder/'settings.json',dict(row=row,freeze_sha256=digest(JOB/'frozen-inputs.json')))
        assert_frozen();dump(folder/'completion.json',parent.seal_files(folder));parent.verify_seal(folder)


def analyze(records):
    if len(records)!=24:return dict(status='pending',passed=False,results=len(records),required=24)
    idx={(r['seed'],r['condition'],r['arm']):r for r in records}
    expected={(s,t,a) for s in SEEDS for t in CONDITIONS for a in ARMS}
    if set(idx)!=expected or len(idx)!=len(records):raise ValueError('missing or duplicate results')
    for s in SEEDS:
        for t in CONDITIONS:
            rows=[idx[s,t,a] for a in ARMS]
            for key in ('parent_model_hash','plan_sha256'):
                if len({r[key] for r in rows})!=1:raise ValueError('paired inputs differ')
    checks=[]
    # Predeclared primary candidate is the full proposal; no post-hoc best-arm selection.
    for s in SEEDS:
        clean=idx[s,'none','anchored_suppress'];attack=idx[s,'model_replacement','anchored_suppress']
        control=idx[s,'model_replacement','anchored_clean'];base=idx[s,'model_replacement','hard_patch']
        v=dict(seed=s,clean_admitted=clean['decision']['passed'],attack_admitted=attack['decision']['passed'],
            clean_delta=clean['committed']['accuracy']-clean['baseline']['accuracy'],
            attacked_delta=attack['committed']['accuracy']-attack['baseline']['accuracy'],
            attacked_vs_clean=attack['committed']['accuracy']-clean['baseline']['accuracy'],
            committed_trigger_rate=attack['committed']['trigger_target_rate'],
            trigger_reduction=attack['baseline']['trigger_target_rate']-attack['committed']['trigger_target_rate'],
            gain_over_raw_clean_anchor=control['raw']['trigger_target_rate']-attack['committed']['trigger_target_rate'],
            gain_over_raw_v19=base['raw']['trigger_target_rate']-attack['committed']['trigger_target_rate'])
        v['passed']=bool(v['clean_admitted'] and v['attack_admitted'] and
            min(v[k] for k in ('clean_delta','attacked_delta','attacked_vs_clean'))>=-.02-1e-12 and
            v['committed_trigger_rate']<=.20+1e-12 and v['trigger_reduction']>=.20-1e-12 and
            min(v['gain_over_raw_clean_anchor'],v['gain_over_raw_v19'])>=.10-1e-12)
        v['aspirational_below_10_percent']=v['committed_trigger_rate']<.10
        checks.append(v)
    passed=all(v['passed'] for v in checks)
    return dict(status='checkpoint_promising' if passed else 'completed_targets_not_met',passed=passed,checks=checks,
        primary_candidate='anchored_suppress',publication_ready=False,automatic_followup=False,end_to_end_evidence=False,
        scope='Six exposed FEMNIST round-100 checkpoints; no unseen triggers or independent confirmation.')


def report():
    records=[]
    for row in manifest():
        folder=OUT/row['id']
        if (folder/'completion.json').exists():
            parent.verify_seal(folder);records.extend(read(folder/'summary.json')['records'])
    verdict=analyze(records);dump(JOB/'analysis.json',verdict)
    dump(JOB/'results.json',dict(records=records,analysis=verdict))
    lines=['# V20 anchored repair diagnostic','',f"Status: {verdict['status']}; cells: {len(records)}/24.",
        'Round-100 exposed FEMNIST checkpoints. Trigger rate is on all non-target-label evaluation examples.',
        'Admission uses unchanged private panels; global accuracy is an offline study target.','',
        '| Condition | Seed | Arm | Admitted | Raw accuracy | Raw trigger rate | Committed accuracy | Committed trigger rate |',
        '|---|---:|---|---|---:|---:|---:|---:|']
    for r in records:
        lines.append(f"| {r['condition']} | {r['seed']} | {r['arm']} | {r['decision']['passed']} | {100*r['raw']['accuracy']:.2f}% | {100*r['raw']['trigger_target_rate']:.2f}% | {100*r['committed']['accuracy']:.2f}% | {100*r['committed']['trigger_target_rate']:.2f}% |")
    lines+=['','All raw/committed conditional ASR denominators, lower-decile client utility, collapse metrics,',
        'private panel decisions, hashes, loss components, teacher work and simulated communication are in the JSON artifacts.',
        'No automatic tuning or full federated training follows this diagnostic.']
    (JOB/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return verdict


def run():
    with exclusive_lock(JOB/'coordinator.lock'):
        assert_frozen();state=dict(status='running',pid=os.getpid(),completed=0,total=6,cells=24,failed=0,
            started_utc=datetime.now(timezone.utc).isoformat(),publication_ready=False)
        dump(JOB/'status.json',state)
        dump(JOB/'environment.json',dict(torch=torch.__version__,cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None))
        try:
            for i,row in enumerate(manifest()):
                assert_frozen();state.update(current=row['id'],updated_utc=datetime.now(timezone.utc).isoformat())
                dump(JOB/'status.json',state);folder=OUT/row['id']
                if (folder/'completion.json').exists():
                    parent.verify_seal(folder)
                    if read(folder/'settings.json')!=dict(row=row,freeze_sha256=digest(JOB/'frozen-inputs.json')):
                        raise ValueError('completed settings mismatch')
                else:
                    if folder.exists():
                        if not folder.resolve().is_relative_to(OUT.resolve()):raise ValueError('invalid partial output path')
                        folder.rename(folder.with_name(folder.name+f'.interrupted-{time.time_ns()}'))
                    print(f"START {i+1}/6 {row['id']}",flush=True)
                    result=subprocess.run([sys.executable,'-u','-m','scripts.trust_anchor_v20.study','--cell',str(i)],cwd=ROOT)
                    if result.returncode:raise RuntimeError(f"worker failed: {row['id']} exit {result.returncode}")
                    parent.verify_seal(folder)
                state['completed']+=1;report();dump(JOB/'status.json',state)
            verdict=report();state.update(status=verdict['status'],finished_utc=datetime.now(timezone.utc).isoformat())
            dump(JOB/'status.json',state)
        except Exception as error:
            state.update(status='failed',failed=1,error=str(error),traceback=traceback.format_exc())
            dump(JOB/'status.json',state);raise


def main():
    parser=argparse.ArgumentParser();action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--freeze',action='store_true');action.add_argument('--run',action='store_true')
    action.add_argument('--cell',type=int);action.add_argument('--report',action='store_true');args=parser.parse_args()
    if args.freeze:
        import json
        value=json.loads(json.dumps(inputs()));path=JOB/'frozen-inputs.json'
        if path.exists() and read(path)!=value:raise ValueError('V20 already frozen differently')
        dump(path,value);print(f'Frozen V20: {digest(path)}')
    elif args.run:run()
    elif args.report:assert_frozen();report()
    else:execute(args.cell)


if __name__=='__main__':main()
