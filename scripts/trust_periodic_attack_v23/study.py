"""Six frozen attack trajectories; reuse verified clean evidence and controls."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_name,'2')
import argparse
from dataclasses import replace
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
from scripts.trust_periodic_v22 import study as base
from scripts.trust_online_v21_fixed.parity import historical_weights
from scripts.trust_repair_v16_fixed.integrity import snapshot,state_hash,assert_hash
from scripts.trust_repair_v16_fixed.research.models import create_model
from scripts.trust_repair_v16_fixed.research.splits import make_splits
from scripts.trust_repair_v16_fixed.runtime import exclusive_lock
from .analysis import SEEDS,ATTACKS,ARMS,clean_gate,qualify

ROOT=base.ROOT
JOB=ROOT/'artifacts/periodic-attack-v23'
OUT=ROOT/'results/periodic-attack-v23'
POLICY_FILE=ROOT/'docs/research/PERIODIC_ATTACK_V23.md'
TEST_FILE=ROOT/'tests/test_periodic_attack_v23.py'
LAUNCHER=ROOT/'scripts/run_periodic_attack_v23.ps1'
read,dump,digest=base.read,base.dump,base.digest


def config(seed,attack):
    if seed not in SEEDS or attack not in ('model_replacement','low_and_slow'):
        raise ValueError('unknown new V23 attack cell')
    return replace(base.config(seed,attack,'v20_periodic'),
                   experiment_id=f'v23-femnist-v20_periodic-{attack}-s{seed}')


def manifest():
    # Targeted evidence is the largest uncertainty; finish all six irrespective
    # of performance outcomes. Only software/integrity failures stop execution.
    return [dict(seed=s,attack=t,arm='v20_periodic',id=config(s,t).experiment_id)
            for t in ('model_replacement','low_and_slow') for s in SEEDS]


def references():
    clean=[dict(seed=s,attack='none',arm=a,kind='v22_clean',
                path=str((base.OUT/base.config(s,'none',a).experiment_id).relative_to(ROOT)))
           for s in SEEDS for a in ARMS]
    controls=[dict(seed=s,attack=t,arm=a,kind='v16_control',
                  path=str(base.source_folder(s,t,a).relative_to(ROOT)))
              for t in ('low_and_slow','model_replacement') for s in SEEDS for a in ('clip_only','current')]
    return clean+controls


def inputs():
    base.assert_frozen()
    previous=read(base.JOB/'status.json')
    if previous['status']!='clean_rejected' or previous['completed']!=12 or previous['failed'] or previous['held']!=24:
        raise ValueError('V22 failed primary result must be complete and preserved')
    refs=[]
    for ref in references():
        path=ROOT/ref['path']
        refs.append(dict(ref,completion_sha256=digest(path/'completion.json'),
                         summary_sha256=digest(path/'summary.json'),trace_sha256=digest(path/'chain-trace.json')))
    return dict(source={str(p.relative_to(ROOT)):digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        v22_freeze_sha256=digest(base.JOB/'frozen-inputs.json'),
        v22_result_sha256=digest(base.JOB/'analysis.json'),references=refs,
        manifest=[dict(r,config=config(r['seed'],r['attack']).to_dict()) for r in manifest()],
        policy_sha256=digest(POLICY_FILE),tests_sha256=digest(TEST_FILE),launcher_sha256=digest(LAUNCHER),
        primary_candidate='v20_periodic',selected_after_exposed_clean_results=True,
        new_trajectories=6,reused_rows=21,automatic_followup=False,publication_ready=False)


def assert_frozen():
    if json.loads(json.dumps(inputs()))!=read(JOB/'frozen-inputs.json'):
        raise ValueError('frozen V23 source or references changed')


def reference_rows():
    rows=[]
    for ref in references():
        folder=ROOT/ref['path']
        if ref['kind']=='v22_clean':
            base.parent.verify_seal(folder)
            row=read(folder/'online-summary.json')
        else:
            if read(folder/'completion.json')!=base.v16.sealed_files(folder):
                raise ValueError('historical comparator seal mismatch')
            summary=read(folder/'summary.json')
            c=base.v16.config('femnist',ref['arm'],ref['seed'],ref['attack'])
            integrity=read(folder/'integrity-summary.json')
            if summary['configuration']!=c.to_dict() or not integrity['passed'] or integrity['checked_rounds']!=100:
                raise ValueError('historical comparator config or integrity mismatch')
            weights=historical_weights(folder,summary)
            summary=dict(summary,accepted_malicious_weight_history=weights,
                         mean_post_onset_malicious_weight=float(np.mean(weights[c.attack_start_round-1:])))
            events=[json.loads(line) for line in (folder/'events.jsonl').open(encoding='utf-8')]
            audit=dict(passed=True,arm=ref['arm'],checked_rounds=100,reused=True,
                       historical_weight_source='reconstructed from recorded accepted IDs and sample counts')
            row=base.extract(summary,[],[],audit,events)
        if (row['seed'],row['attack'],row['arm'])!=(ref['seed'],ref['attack'],ref['arm']):
            raise ValueError('reused evidence identity mismatch')
        rows.append(dict(row,reused=True,source_path=ref['path'],
                         source_completion_sha256=digest(folder/'completion.json')))
    gate=clean_gate(rows)
    if not gate['passed']:raise ValueError('promoted periodic arm does not satisfy the predeclared clean gate')
    return rows


def collect_rows():
    rows=reference_rows()
    for row in manifest():
        folder=OUT/row['id']
        if (folder/'completion.json').exists():
            base.parent.verify_seal(folder)
            result=read(folder/'online-summary.json')
            if result['reused']:raise ValueError('new trajectory mislabeled reused')
            rows.append(result)
    return rows


def execute(index):
    with exclusive_lock(JOB/'cell.lock'),threadpool_limits(limits=2):
        assert_frozen();reference_rows()
        row=manifest()[index];c=config(row['seed'],row['attack']);folder=OUT/row['id']
        if folder.exists():raise FileExistsError('refusing existing V23 output')
        torch.set_num_threads(2)
        if not torch.cuda.is_available():raise RuntimeError('CUDA required for historical model parity')
        reference=base.source_folder(row['seed'],row['attack'],'current')
        x,y,tx,ty=base.parent.load_arrays(row['seed']);split=make_splits(y,c,external_test=True)
        runner=base.BenchmarkRunner(c,folder);runner.challenge_arm='current'
        runner.logger=base.ProgressLogger(runner.logger,folder,row['id'])
        runner.online_controller=base.OnlineRepair('v20_periodic',base.DEPLOYMENT,base.v20.POLICY,base.v20.ANCHOR)
        observer=base.OfflineObserver(tx,ty,c.backdoor_target_label,split.natural_test_clients)
        runner.offline_observer=observer
        runner.run_arrays(x,y,lambda:create_model(c.dataset,x.shape[1:],c.num_classes,c.model),(tx,ty))
        parity=base.validate_parity(folder,reference,'v20_periodic');dump(folder/'parity.json',parity)
        state=snapshot(runner.final_model);final_hash=state_hash(state)
        trace=base.trace_hashes(folder)
        if final_hash!=trace[-1]:raise ValueError('final state differs from committed trace')
        torch.save(dict(model_hash=final_hash,state=state),folder/'final.pt')
        events=[json.loads(line) for line in (folder/'events.jsonl').open(encoding='utf-8')]
        integrity=[e for e in events if e['event_type']=='model_integrity']
        if [e['round_index'] for e in integrity]!=list(range(1,101)) or not all(
            e['payload']['passed'] and e['payload']['expected_hash']==e['payload']['committed_hash'] for e in integrity):
            raise ValueError('missing global integrity checks')
        online=[e['payload'] for e in events if e['event_type']=='online_repair']
        monitors=[e['payload'] for e in events if e['event_type']=='private_reference_monitor']
        if ([r['round'] for r in online]!=[25,50,75,100] or
            [r['round'] for r in observer.records]!=[25,50,75,100] or
            [r['round'] for r in monitors]!=list(range(1,101))):
            raise ValueError('fixed schedule or monitoring incomplete')
        for r in online:
            if r['committed_model_hash']!=trace[r['round']]:raise ValueError('repair selection differs from committed trace')
            if r['decision']['memory_gate_applied'] or r['decision']['passed']!=r['decision']['base_verifier_passed']:
                raise ValueError('periodic-only admission rule changed')
        dump(folder/'offline-repair-evaluation.json',dict(records=observer.records))
        audit=dict(passed=True,arm='v20_periodic',checked_rounds=100,online_decisions=4,
                   final_model_hash=final_hash,parity=parity)
        dump(folder/'integrity-summary.json',audit)
        summary=base.extract(read(folder/'summary.json'),online,observer.records,audit,events)
        final_eval=base.parent.evaluate(runner.final_model,tx,ty,c.backdoor_target_label,split.natural_test_clients)
        if final_eval['model_hash']!=final_hash or final_eval['accuracy']!=summary['final_accuracy']:
            raise ValueError('final offline evaluation mismatch')
        dump(folder/'final-evaluation.json',final_eval)
        summary.update(final_evaluation=final_eval,reused=False,source_path=str(folder.relative_to(ROOT)),
            private_reference_presentations=sum(m['example_presentations'] for m in monitors),
            private_reference_transmitted_bytes=sum(m['extra_transmitted_bytes'] for m in monitors),
            memory_candidate_presentations=sum(r['memory_candidate_presentations'] for r in online),
            private_reference_seconds=sum(m['seconds'] for m in monitors))
        summary['extra_repair_example_presentations']+=summary['private_reference_presentations']+summary['memory_candidate_presentations']
        summary['extra_repair_transmitted_bytes']+=summary['private_reference_transmitted_bytes']
        if row['attack']=='model_replacement':
            history=read(folder/'summary.json')['targeted_attack_success_history']
            summary['reinfection']=[dict(round=r['round'],pre_repair_trigger_rate=r['source']['trigger_target_rate'],
                previous_committed_trigger_rate=history[r['round']-2],
                increase=r['source']['trigger_target_rate']-history[r['round']-2]) for r in observer.records]
        dump(folder/'online-summary.json',summary)
        dump(folder/'settings.json',dict(row=row,freeze_sha256=digest(JOB/'frozen-inputs.json')))
        assert_frozen();dump(folder/'completion.json',base.parent.seal_files(folder));base.parent.verify_seal(folder)


def report():
    rows=collect_rows();verdict=qualify(rows)
    dump(JOB/'analysis.json',verdict);dump(JOB/'results.json',dict(records=rows,analysis=verdict))
    new=sum(not r['reused'] for r in rows)
    lines=['# V23 periodic-only V20 attack study','',f"Status: {verdict['status']}; new trajectories: {new}/6; reused comparison rows: 21.",
        'Candidate promoted after exposed V22 clean results. V22 remains clean_rejected with its 24 attack cells held.',
        'Repair rounds 25, 50, 75, 100; original verifier; passive private monitoring retained and costed.','',
        '| Condition | Seed | Arm | Evidence | Final accuracy | Mean post-onset ASR | Endpoint ASR | Admissions/attempts |',
        '|---|---:|---|---|---:|---:|---:|---:|']
    for r in rows:
        rate=lambda k:'n/a' if r[k] is None else f'{100*r[k]:.2f}%'
        lines.append(f"| {r['attack']} | {r['seed']} | {r['arm']} | {'reused' if r['reused'] else 'new'} | {100*r['final_accuracy']:.2f}% | {rate('mean_post_onset_asr')} | {rate('endpoint_asr')} | {r['repair_admissions']}/{r['repair_attempts']} |")
    lines+=['','All targeted post-onset means include rounds 11-100, including time between repairs and holds.',
        'Reused controls are not new independent runs. Conditional ASR/denominators are available for newly evaluated',
        'candidate states; historical controls retain their recorded metric coverage. Full trajectories, lower-tail utility,',
        'honest exclusion, accepted malicious aggregation weight and all monitoring/teacher/repair costs are retained.',
        'No automatic tuning or confirmation. Exposed development data and an in-bank trigger limit claims.']
    (JOB/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return verdict


def audit():
    rows=reference_rows()
    dump(JOB/'reference-audit.json',dict(passed=True,rows=len(rows),reused_clean_rows=9,reused_attack_controls=12,
        clean_gate=clean_gate(rows),references=[dict(path=r['source_path'],completion=r['source_completion_sha256']) for r in rows],
        new_trajectories_counted_as_complete=0,created_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(dict(passed=True,reused_rows=len(rows),clean_gate=clean_gate(rows)),indent=2))


def run():
    with exclusive_lock(JOB/'coordinator.lock'):
        assert_frozen();reference_rows()
        if not read(JOB/'validation.json')['passed'] or not read(JOB/'reference-audit.json')['passed']:
            raise ValueError('tests and reference audit required')
        state=dict(status='running',pid=os.getpid(),completed=0,total=6,reused_rows=21,failed=0,
                   started_utc=datetime.now(timezone.utc).isoformat(),publication_ready=False)
        dump(JOB/'status.json',state)
        dump(JOB/'environment.json',dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0)))
        try:
            report()
            for i,row in enumerate(manifest()):
                assert_frozen();state.update(current=row['id'],updated_utc=datetime.now(timezone.utc).isoformat())
                dump(JOB/'status.json',state);folder=OUT/row['id']
                if (folder/'completion.json').exists():
                    base.parent.verify_seal(folder)
                    if read(folder/'settings.json')!=dict(row=row,freeze_sha256=digest(JOB/'frozen-inputs.json')):
                        raise ValueError('completed settings mismatch')
                else:
                    if folder.exists():
                        if not folder.resolve().is_relative_to(OUT.resolve()):raise ValueError('invalid partial output path')
                        folder.rename(folder.with_name(folder.name+f'.interrupted-{time.time_ns()}'))
                    print(f"START {i+1}/6 {row['id']}",flush=True)
                    result=subprocess.run([sys.executable,'-u','-m','scripts.trust_periodic_attack_v23.study','--cell',str(i)],cwd=ROOT)
                    if result.returncode:raise RuntimeError(f"worker failed {row['id']}: {result.returncode}")
                    base.parent.verify_seal(folder)
                state['completed']+=1;verdict=report();dump(JOB/'status.json',state)
                print(f"DONE {i+1}/6 {row['id']}",flush=True)
            state.update(status=verdict['status'],finished_utc=datetime.now(timezone.utc).isoformat())
            dump(JOB/'status.json',state)
        except Exception as error:
            state.update(status='failed',failed=1,error=str(error),traceback=traceback.format_exc())
            dump(JOB/'status.json',state);raise


def main():
    parser=argparse.ArgumentParser();action=parser.add_mutually_exclusive_group(required=True)
    for option in ('freeze','audit','run','report'):action.add_argument('--'+option,action='store_true')
    action.add_argument('--cell',type=int);args=parser.parse_args()
    if args.freeze:
        value=json.loads(json.dumps(inputs()));path=JOB/'frozen-inputs.json'
        if path.exists() and read(path)!=value:raise ValueError('V23 already frozen differently')
        dump(path,value);print(f'Frozen V23: {digest(path)}')
    elif args.audit:audit()
    elif args.run:run()
    elif args.report:assert_frozen();report()
    else:execute(args.cell)


if __name__=='__main__':main()
