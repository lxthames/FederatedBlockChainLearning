"""Frozen scheduling x objective comparison. Independent confirmation is future work."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_name,'2')

import argparse
from dataclasses import asdict
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

from scripts.trust_supervised_v18 import study as parent
from scripts.trust_supervised_v18.repair import allocate,partition_binding,verify
from scripts.trust_repair_v16_fixed.integrity import snapshot,state_hash,assert_hash
from scripts.trust_repair_v16_fixed.research.models import create_model
from scripts.trust_repair_v16_fixed.research.splits import make_splits
from scripts.trust_repair_v16_fixed.runtime import exclusive_lock
from .repair import OBJECTIVES,SCHEDULES,make_plan,plan_hash,repair_model

ROOT=Path(__file__).resolve().parents[2]
JOB=ROOT/'artifacts/interleave-v19'
OUT=ROOT/'results/interleave-v19'
POLICY_FILE=ROOT/'docs/research/INTERLEAVE_V19.md'
TEST_FILE=ROOT/'tests/test_interleave_v19.py'
LAUNCHER=ROOT/'scripts/run_interleave_v19.ps1'
SEEDS=parent.SEEDS
CONDITIONS=parent.CONDITIONS
POLICY=parent.POLICY
read,dump,digest=parent.read,parent.dump,parent.digest


def manifest():
    return [dict(schedule=schedule,condition=t,seed=s,checkpoint=100,
                 id=f'v19-femnist-{schedule}-{t}-s{s}-r100')
            for schedule in SCHEDULES for t in CONDITIONS for s in SEEDS]


def reference_folder(seed,condition):
    return parent.OUT/f'v18-repair-femnist-{condition}-s{seed}-r100'


def inputs():
    parent.assert_frozen()
    state=read(parent.JOB/'status.json')
    if state['completed']!=24 or state['failed'] or state['status'] not in ('checkpoint_rejected','checkpoint_promising'):
        raise ValueError('V18 must finish without execution failures')
    refs={}
    for t in CONDITIONS:
        for s in SEEDS:
            folder=reference_folder(s,t)
            refs[str(folder.relative_to(ROOT))]=digest(folder/'completion.json')
            replay=parent.OUT/parent.replay_config(s,t).experiment_id
            refs[str(replay.relative_to(ROOT))]=digest(replay/'completion.json')
    return dict(source={str(p.relative_to(ROOT)):digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        parent_freeze_sha256=digest(parent.JOB/'frozen-inputs.json'),references=refs,
        manifest=manifest(),policy=asdict(POLICY),policy_sha256=digest(POLICY_FILE),
        test_sha256=digest(TEST_FILE),launcher_sha256=digest(LAUNCHER),
        automatic_followup=False,publication_ready=False)


def assert_frozen():
    if inputs()!=read(JOB/'frozen-inputs.json'):
        raise ValueError('frozen V19 inputs changed')


def baseline_parity(telemetry,record,reference):
    """Independent V18 implementation must match every optimizer outcome exactly."""
    old=read(reference/f"{record['objective']}-repair.json")
    prior=read(reference/f"{record['objective']}-result.json")
    for key in ('model_hash','policy','steps'):
        if telemetry[key]!=old[key]:
            raise ValueError(f'sequential V18 parity mismatch: {key}')
    if {k:telemetry['work'][k] for k in old['work']}!=old['work']:
        raise ValueError('sequential V18 work mismatch')
    for key in ('raw','committed','decision','held','parent_model_hash'):
        if record[key]!=prior[key]:
            raise ValueError(f'sequential V18 evaluation parity mismatch: {key}')
    return dict(passed=True,checked_optimizer_steps=len(old['steps']),
                matched_candidate_hash=old['model_hash'],reference_completion_sha256=digest(reference/'completion.json'))


def require_baselines():
    for row in manifest()[:6]:
        folder=OUT/row['id']
        parent.verify_seal(folder)
        summary=read(folder/'summary.json')
        if not all(r['parity']['passed'] for r in summary['records']):
            raise ValueError('all six sequential jobs must match V18 before interleaving')


def execute(index):
    with exclusive_lock(JOB/'cell.lock'),threadpool_limits(limits=2):
        assert_frozen()
        row=manifest()[index]
        if row['schedule']=='interleaved':
            require_baselines()
        folder=OUT/row['id']
        if folder.exists():
            raise FileExistsError('refusing existing output folder')
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required for exact historical baseline parity')
        seed,condition=row['seed'],row['condition']
        c=parent.replay_config(seed,condition)
        reference=reference_folder(seed,condition)
        replay=parent.OUT/c.experiment_id
        parent.verify_seal(reference);parent.verify_seal(replay)
        x,y,tx,ty=parent.load_arrays(seed)
        split=make_splits(y,c,external_test=True)
        source=read(replay/'summary.json')
        groups=allocate(split.validation,source['participants'][99],seed,100,POLICY)
        binding=partition_binding(groups)
        if binding!=read(reference/'partitions.json')['binding']:
            raise ValueError('repair/verification split differs from V18')
        factory=lambda:create_model(c.dataset,x.shape[1:],c.num_classes,c.model).to('cuda')
        model,checkpoint=parent.load_checkpoint(replay/'checkpoints'/'round-100.pt',factory)
        local_data={client:(x[ids].copy(),y[ids].copy()) for client,ids in groups['repair'].items()}
        plan=make_plan(local_data,seed,100,POLICY)
        folder.mkdir(parents=True)
        dump(folder/'plan.json',dict(updates=plan,sha256=plan_hash(plan)))
        dump(folder/'partitions.json',read(reference/'partitions.json'))
        before={stage:{int(client):m for client,m in members.items()}
                for stage,members in read(reference/'parent-verification.json').items()}
        if any(m['model_hash']!=checkpoint['model_hash'] for members in before.values() for m in members.values()):
            raise ValueError('parent verification and checkpoint disagree')
        baseline=read(reference/'unchanged-result.json')['raw']
        records=[]
        for objective in OBJECTIVES:
            print(f"{row['id']}: {objective}",flush=True)
            started=time.perf_counter()
            candidate,telemetry=repair_model(model,local_data,objective,row['schedule'],seed,100,POLICY,plan)
            measured=parent.panel_measurements(candidate,groups,x,y)
            verdict=verify(before,measured,POLICY)
            committed=parent.select_committed(model,candidate,verdict)
            raw=parent.evaluate(candidate,tx,ty,c.backdoor_target_label,split.natural_test_clients)
            score=raw if verdict['passed'] else baseline
            assert_hash(committed,score['model_hash'],'scored and committed equality')
            record=dict(seed=seed,condition=condition,checkpoint=100,schedule=row['schedule'],objective=objective,
                parent_model_hash=checkpoint['model_hash'],decision=verdict,raw=raw,committed=score,
                baseline=baseline,held=not verdict['passed'],work=telemetry['work'],
                seconds=time.perf_counter()-started,plan_sha256=telemetry['plan_sha256'])
            record['parity']=baseline_parity(telemetry,record,reference) if row['schedule']=='sequential' else dict(applicable=False)
            dump(folder/f'{objective}-verification.json',measured)
            dump(folder/f'{objective}-repair.json',telemetry)
            dump(folder/f'{objective}-result.json',record)
            torch.save(dict(model_hash=telemetry['model_hash'],state=snapshot(candidate)),folder/f'{objective}-candidate.pt')
            torch.save(dict(model_hash=score['model_hash'],state=snapshot(committed)),folder/f'{objective}-committed.pt')
            records.append(record)
        if records[0]['work']!=records[1]['work']:
            raise ValueError('objectives differ in work budget within schedule')
        assert_hash(model,checkpoint['model_hash'],'parent preserved')
        verification_work=dict(read(reference/'summary.json')['verification_work'],
                               evaluated_models=2,reused_parent_measurements=1)
        dump(folder/'summary.json',dict(row=row,records=records,repair_budgets_matched=True,
            verification_work=verification_work,
            note='Same inherited per-model verification work; two candidates plus reused parent measurements.'))
        dump(folder/'settings.json',dict(row=row,policy=asdict(POLICY),freeze_sha256=digest(JOB/'frozen-inputs.json')))
        assert_frozen()
        dump(folder/'completion.json',parent.seal_files(folder));parent.verify_seal(folder)


def analyze(records):
    if len(records)!=24:
        return dict(status='pending',passed=False,results=len(records),required=24)
    expected={(s,t,schedule,o) for s in SEEDS for t in CONDITIONS for schedule in SCHEDULES for o in OBJECTIVES}
    index={(r['seed'],r['condition'],r['schedule'],r['objective']):r for r in records}
    if set(index)!=expected or len(records)!=len(index):
        raise ValueError('missing or duplicate factorial results')
    paired=[];interactions=[]
    budget_keys=('optimizer_steps','training_examples','proposal_examples','diagnostic_examples')
    for seed in SEEDS:
        for condition in CONDITIONS:
            effects={}
            for objective in OBJECTIVES:
                a=index[seed,condition,'sequential',objective];b=index[seed,condition,'interleaved',objective]
                if a['plan_sha256']!=b['plan_sha256'] or a['parent_model_hash']!=b['parent_model_hash']:
                    raise ValueError('paired treatment inputs differ')
                if any(a['work'][k]!=b['work'][k] for k in budget_keys):
                    raise ValueError('training/diagnostic work differs across schedules')
                effect=dict(seed=seed,condition=condition,objective=objective,
                    raw_accuracy_gain=b['raw']['accuracy']-a['raw']['accuracy'],
                    raw_trigger_rate_reduction=a['raw']['trigger_target_rate']-b['raw']['trigger_target_rate'],
                    committed_accuracy_gain=b['committed']['accuracy']-a['committed']['accuracy'],
                    committed_trigger_rate_reduction=a['committed']['trigger_target_rate']-b['committed']['trigger_target_rate'])
                paired.append(effect);effects[objective]=effect
            interactions.append(dict(seed=seed,condition=condition,
                accuracy_schedule_by_objective=effects['hard_patch']['raw_accuracy_gain']-effects['clean_tune']['raw_accuracy_gain'],
                trigger_schedule_by_objective=effects['hard_patch']['raw_trigger_rate_reduction']-effects['clean_tune']['raw_trigger_rate_reduction']))
    checks=[]
    for seed in SEEDS:
        clean=index[seed,'none','interleaved','hard_patch']
        attacked=index[seed,'model_replacement','interleaved','hard_patch']
        tune=index[seed,'model_replacement','interleaved','clean_tune']
        reference=reference_folder(seed,'model_replacement')
        parent.verify_seal(reference)
        prior_random=read(reference/'random_patch-result.json')
        values=dict(seed=seed,clean_admitted=clean['decision']['passed'],attack_admitted=attacked['decision']['passed'],
            clean_accuracy_delta=clean['committed']['accuracy']-clean['baseline']['accuracy'],
            attacked_accuracy_delta=attacked['committed']['accuracy']-attacked['baseline']['accuracy'],
            attacked_accuracy_vs_clean=attacked['committed']['accuracy']-clean['baseline']['accuracy'],
            asr=attacked['committed']['trigger_target_rate'],
            asr_reduction=attacked['baseline']['trigger_target_rate']-attacked['committed']['trigger_target_rate'],
            gain_over_interleaved_clean_tune=tune['raw']['trigger_target_rate']-attacked['committed']['trigger_target_rate'],
            gain_over_v18_random_patch=prior_random['raw']['trigger_target_rate']-attacked['committed']['trigger_target_rate'])
        passed=(values['clean_admitted'] and values['attack_admitted'] and
            min(values[k] for k in ('clean_accuracy_delta','attacked_accuracy_delta','attacked_accuracy_vs_clean'))>=-.02-1e-12 and
            values['asr']<=.20+1e-12 and values['asr_reduction']>=.20-1e-12 and
            values['gain_over_interleaved_clean_tune']>=.10-1e-12 and values['gain_over_v18_random_patch']>=.10-1e-12)
        checks.append(dict(values,passed=bool(passed)))
    passed=all(c['passed'] for c in checks)
    return dict(status='checkpoint_promising' if passed else 'completed_targets_not_met',passed=passed,
        paired_effects=paired,interactions=interactions,checks=checks,publication_ready=False,
        automatic_followup=False,end_to_end_evidence=False,
        scope='round-100 exposed-seed scheduling diagnostic; no cause is established by a target pass alone')


def report():
    records=[]
    for row in manifest():
        folder=OUT/row['id']
        if (folder/'completion.json').exists():
            parent.verify_seal(folder)
            records.extend(read(folder/'summary.json')['records'])
    verdict=analyze(records)
    dump(JOB/'analysis.json',verdict);dump(JOB/'results.json',dict(records=records,analysis=verdict))
    lines=['# V19 interleaving diagnostic','',f"Status: {verdict['status']}; factorial cells: {len(records)}/24.",
        'Fixed round-100 FEMNIST development checkpoints. Raw and committed outcomes are distinct.','',
        '| Condition | Seed | Schedule | Objective | Admitted | Raw accuracy | Raw trigger rate | Committed accuracy | Committed trigger rate |',
        '|---|---:|---|---|---|---:|---:|---:|---:|']
    for r in records:
        lines.append(f"| {r['condition']} | {r['seed']} | {r['schedule']} | {r['objective']} | {r['decision']['passed']} | {100*r['raw']['accuracy']:.2f}% | {100*r['raw']['trigger_target_rate']:.2f}% | {100*r['committed']['accuracy']:.2f}% | {100*r['committed']['trigger_target_rate']:.2f}% |")
    if verdict['status']!='pending':
        lines+=['','| Condition | Seed | Objective | Raw accuracy gain from interleaving | Raw trigger-rate reduction |',
                '|---|---:|---|---:|---:|']
        for r in verdict['paired_effects']:
            lines.append(f"| {r['condition']} | {r['seed']} | {r['objective']} | {100*r['raw_accuracy_gain']:+.2f}pp | {100*r['raw_trigger_rate_reduction']:+.2f}pp |")
    (JOB/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return verdict


def run():
    with exclusive_lock(JOB/'coordinator.lock'):
        assert_frozen()
        state=dict(status='running',pid=os.getpid(),completed=0,total=12,factorial_cells=24,failed=0,
                   started_utc=datetime.now(timezone.utc).isoformat(),publication_ready=False)
        dump(JOB/'status.json',state)
        dump(JOB/'environment.json',dict(torch=torch.__version__,cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None))
        try:
            for i,row in enumerate(manifest()):
                assert_frozen();state.update(current=row['id'],phase=row['schedule'],updated_utc=datetime.now(timezone.utc).isoformat())
                dump(JOB/'status.json',state);folder=OUT/row['id']
                if (folder/'completion.json').exists():
                    parent.verify_seal(folder)
                    if read(folder/'settings.json')!=dict(row=row,policy=asdict(POLICY),freeze_sha256=digest(JOB/'frozen-inputs.json')):
                        raise ValueError('completed settings mismatch')
                else:
                    if folder.exists():
                        if not folder.resolve().is_relative_to(OUT.resolve()):
                            raise ValueError('partial job outside output root')
                        folder.rename(folder.with_name(folder.name+f'.interrupted-{time.time_ns()}'))
                    print(f"START {i+1}/12 {row['id']}",flush=True)
                    result=subprocess.run([sys.executable,'-u','-m','scripts.trust_interleave_v19.study','--cell',str(i)],cwd=ROOT)
                    if result.returncode:
                        raise RuntimeError(f"worker failed: {row['id']} exit {result.returncode}")
                    parent.verify_seal(folder)
                state['completed']+=1;report();dump(JOB/'status.json',state)
                print(f"DONE {i+1}/12 {row['id']}",flush=True)
            verdict=report();state.update(status=verdict['status'],finished_utc=datetime.now(timezone.utc).isoformat())
            dump(JOB/'status.json',state)
        except Exception as error:
            state.update(status='failed',failed=1,error=str(error),traceback=traceback.format_exc())
            dump(JOB/'status.json',state);raise


def main():
    parser=argparse.ArgumentParser();action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--freeze',action='store_true');action.add_argument('--run',action='store_true')
    action.add_argument('--cell',type=int);action.add_argument('--report',action='store_true')
    args=parser.parse_args()
    if args.freeze:
        value=inputs();path=JOB/'frozen-inputs.json'
        if path.exists() and read(path)!=value:
            raise ValueError('V19 already frozen differently')
        dump(path,value);print(f'Frozen V19: {digest(path)}')
    elif args.run:run()
    elif args.report:report()
    else:execute(args.cell)


if __name__=='__main__':
    main()
