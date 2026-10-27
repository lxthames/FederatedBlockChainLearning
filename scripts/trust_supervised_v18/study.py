"""Sealed checkpoint-repair feasibility experiment; no automatic defense promotion."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_name, '2')

import argparse
import copy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from scripts.trust_alignment_v17 import study as v17
from scripts.trust_repair_v16_fixed import study as v16
from scripts.trust_repair_v16_fixed.integrity import snapshot, state_hash, assert_hash
from scripts.trust_repair_v16_fixed.research.models import create_model
from scripts.trust_repair_v16_fixed.research.splits import make_splits
from scripts.trust_repair_v16_fixed.runtime import exclusive_lock
from .capture import capture
from .repair import ARMS, STAGES, RepairPolicy, allocate, partition_binding, checked_load, repair_model, measure, verify

ROOT = Path(__file__).resolve().parents[2]
JOB = ROOT/'artifacts/supervised-v18'
OUT = ROOT/'results/supervised-v18'
POLICY_FILE = ROOT/'docs/research/SUPERVISED_V18.md'
TEST_FILE = ROOT/'tests/test_supervised_v18.py'
LAUNCHER = ROOT/'scripts/run_supervised_v18.ps1'
SEEDS = (11, 17, 23)
CHECKPOINTS = (11, 30, 100)
CONDITIONS = ('none', 'model_replacement')
POLICY = RepairPolicy()
read, dump, digest = v17.read, v17.dump, v17.digest


def replay_config(seed, condition):
    c = v17.config('femnist', 'current', seed, condition, passive=True)
    # Retain the frozen runner's validated protocol/config class; the V18
    # experiment ID and sealed settings identify this passive observer study.
    return replace(c, experiment_id=f'v18-replay-femnist-current-{condition}-s{seed}')


def manifest():
    replay = [dict(role='replay', seed=s, condition=t,
                   id=replay_config(s,t).experiment_id) for t in CONDITIONS for s in SEEDS]
    repairs = [dict(role='repair', seed=s, condition=t, checkpoint=r,
                    id=f'v18-repair-femnist-{t}-s{s}-r{r:03}')
               for t in CONDITIONS for s in SEEDS for r in CHECKPOINTS]
    return replay+repairs


def source_reference(seed, condition):
    c = v16.config('femnist', 'current', seed, condition)
    return v16.OUT/c.experiment_id


def inputs():
    if v17.inputs() != read(v17.JOB/'frozen-inputs.json'):
        raise ValueError('frozen historical source/data changed')
    references = {}
    for t in CONDITIONS:
        for s in SEEDS:
            folder = source_reference(s,t)
            references[folder.name] = dict(completion=digest(folder/'completion.json'),
                trace=digest(folder/'chain-trace.json'), summary=digest(folder/'summary.json'))
    return dict(source={str(p.relative_to(ROOT)):digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
                historical_freeze=digest(v17.JOB/'frozen-inputs.json'),
                data=read(v17.JOB/'frozen-inputs.json')['data'], references=references,
                policy=asdict(POLICY), manifest=manifest(), checkpoints=list(CHECKPOINTS),
                policy_sha256=digest(POLICY_FILE), tests_sha256=digest(TEST_FILE),
                launcher_sha256=digest(LAUNCHER), automatic_end_to_end=False,
                publication_ready=False)


def assert_frozen():
    if inputs() != read(JOB/'frozen-inputs.json'):
        raise ValueError('frozen V18 inputs changed')


def seal_files(folder):
    return {str(p.relative_to(folder)):digest(p) for p in sorted(folder.rglob('*'))
            if p.is_file() and p.name != 'completion.json'}


def verify_seal(folder):
    if read(folder/'completion.json') != seal_files(folder):
        raise ValueError(f'completion seal mismatch: {folder}')


def load_arrays(seed):
    data = v16.DATA/f'femnist-{seed}'
    return [np.load(data/f'{name}.npy', allow_pickle=False) for name in
            ('features', 'labels', 'development_features', 'development_labels')]


def load_checkpoint(path, factory):
    value = torch.load(path, map_location='cpu', weights_only=True)
    if state_hash(value['state']) != value['model_hash']:
        raise ValueError('checkpoint model hash mismatch')
    model = factory()
    checked_load(model, value['state'], 'load captured checkpoint')
    return model, value


def run_replay(row, folder):
    c = replay_config(row['seed'], row['condition'])
    reference = source_reference(row['seed'], row['condition'])
    # Validate every recorded file in this matched historical cell.
    if read(reference/'completion.json') != v16.sealed_files(reference):
        raise ValueError('historical reference seal mismatch')
    x,y,tx,ty = load_arrays(row['seed'])
    hashes = capture(c, folder, x,y,
        lambda:create_model(c.dataset,x.shape[1:],c.num_classes,c.model), (tx,ty), CHECKPOINTS)
    parity = v17.validate_passive_parity(folder, reference)
    events = [json.loads(line) for line in (folder/'events.jsonl').open(encoding='utf-8')]
    integrity = [e for e in events if e['event_type'] == 'model_integrity']
    if [e['round_index'] for e in integrity] != list(range(1,101)) or not all(
            e['payload']['passed'] and e['payload']['expected_hash'] == e['payload']['committed_hash'] for e in integrity):
        raise ValueError('replay integrity checks incomplete')
    trace = read(folder/'chain-trace.json')
    if any(hashes[r] != trace[r]['model_hash'] for r in CHECKPOINTS):
        raise ValueError('checkpoint and matched trajectory differ')
    dump(folder/'parity.json', parity)
    dump(folder/'capture.json', dict(hashes=hashes, checked_models=101, integrity_rounds=100,
                                   condition=row['condition'], seed=row['seed']))


def evaluate(model, features, labels, target, test_clients, batch=256):
    """Offline evaluation only. Never called by repair or admission logic."""
    expected = state_hash(snapshot(model))
    device = next(model.parameters()).device
    model.eval()
    clean, triggered = [], []
    with torch.no_grad():
        for start in range(0,len(labels),batch):
            x = torch.as_tensor(features[start:start+batch], dtype=torch.float32, device=device)
            clean.extend(model(x).argmax(1).cpu().tolist())
            x = x.clone(); x[:,:,-2:,-2:] = 0.
            triggered.extend(model(x).argmax(1).cpu().tolist())
    clean, triggered = np.asarray(clean), np.asarray(triggered)
    mask = labels != target
    correct = (clean == labels)
    conditional = mask & correct
    clients = [float(correct[ids].mean()) for ids in test_clients if len(ids)]
    assert_hash(model, expected, 'offline evaluation')
    return dict(model_hash=expected, accuracy=float(correct.mean()),
                trigger_target_rate=float((triggered[mask] == target).mean()),
                clean_target_rate=float((clean[mask] == target).mean()),
                conditional_asr=float((triggered[conditional] == target).mean()) if conditional.any() else None,
                conditional_asr_denominator=int(conditional.sum()),
                lower_decile_client_utility=float(np.quantile(clients,.1)),
                client_utilities=clients, maximum_prediction_fraction=float(np.bincount(clean).max()/len(clean)))


def panel_measurements(model, groups, x, y):
    return {stage:{client:measure(model,x[ids],y[ids]) for client,ids in groups[stage].items()}
            for stage in STAGES}


def select_committed(parent, candidate, verdict):
    candidate_hash = state_hash(snapshot(candidate))
    if candidate_hash != verdict['scored_model_hash']:
        raise ValueError('scored candidate changed before commitment')
    state = snapshot(candidate if verdict['passed'] else parent)
    committed = copy.deepcopy(parent)
    checked_load(committed, state, 'verified repair commitment')
    return committed


def run_repair(row, folder):
    started = time.perf_counter()
    seed, condition, rnd = row['seed'], row['condition'], row['checkpoint']
    c = replay_config(seed, condition)
    replay_folder = OUT/c.experiment_id
    verify_seal(replay_folder)
    x,y,tx,ty = load_arrays(seed)
    split = make_splits(y,c,external_test=True)
    summary = read(replay_folder/'summary.json')
    groups = allocate(split.validation,summary['participants'][rnd-1],seed,rnd,POLICY)
    factory = lambda:create_model(c.dataset,x.shape[1:],c.num_classes,c.model).to('cuda')
    parent, checkpoint = load_checkpoint(replay_folder/'checkpoints'/f'round-{rnd:03}.pt',factory)
    if checkpoint['round'] != rnd:
        raise ValueError('wrong checkpoint round')
    folder.mkdir(parents=True,exist_ok=False)
    dump(folder/'partitions.json', dict(groups={stage:{str(k):v.tolist() for k,v in members.items()}
                                              for stage,members in groups.items()},binding=partition_binding(groups),
                                     replay_completion_sha256=digest(replay_folder/'completion.json')))
    # Repair gets only this subset; verification and evaluation arrays are withheld.
    local_data = {client:(x[ids].copy(),y[ids].copy()) for client,ids in groups['repair'].items()}
    parent_measurements = panel_measurements(parent,groups,x,y)
    dump(folder/'parent-verification.json',parent_measurements)
    reference_score = evaluate(parent,tx,ty,c.backdoor_target_label,split.natural_test_clients)
    records = []
    for arm in ARMS:
        print(f"{row['id']}: {arm}",flush=True)
        start = time.perf_counter()
        candidate, telemetry = repair_model(parent,local_data,arm,seed,rnd,POLICY)
        measured = parent_measurements if arm == 'unchanged' else panel_measurements(candidate,groups,x,y)
        verdict = verify(parent_measurements,measured,POLICY)
        committed = select_committed(parent,candidate,verdict)
        # No test outcomes have been observed before the admission decision.
        raw_score = reference_score if arm == 'unchanged' else evaluate(candidate,tx,ty,c.backdoor_target_label,split.natural_test_clients)
        committed_score = raw_score if verdict['passed'] or arm == 'unchanged' else reference_score
        assert_hash(committed,committed_score['model_hash'],'scored committed-model equality')
        record = dict(arm=arm,seed=seed,condition=condition,checkpoint=rnd,
                      parent_model_hash=checkpoint['model_hash'],decision=verdict,
                      raw=raw_score,committed=committed_score,work=telemetry['work'],
                      seconds=time.perf_counter()-start,
                      held=bool(arm != 'unchanged' and not verdict['passed']))
        dump(folder/f'{arm}-verification.json',measured)
        dump(folder/f'{arm}-repair.json',telemetry)
        dump(folder/f'{arm}-result.json',record)
        torch.save(dict(model_hash=telemetry['model_hash'],state=snapshot(candidate)),folder/f'{arm}-candidate.pt')
        torch.save(dict(model_hash=committed_score['model_hash'],state=snapshot(committed)),folder/f'{arm}-committed.pt')
        records.append(record)
    budgets = [r['work'] for r in records if r['arm'] != 'unchanged']
    if any(b != budgets[0] for b in budgets):
        raise ValueError('repair controls have different work budgets')
    assert_hash(parent,checkpoint['model_hash'],'parent preserved through all repair arms')
    model_bytes = sum(v.numel()*v.element_size() for v in parent.state_dict().values())
    evaluations = sum(m['example_evaluations'] for group in parent_measurements.values() for m in group.values())
    dump(folder/'summary.json',dict(row=row,records=records,passed_software_checks=True,
         repair_budgets_matched=True,seconds=time.perf_counter()-started,
         verification_work=dict(example_evaluations_per_model=evaluations,
             evaluated_models=4,model_transmissions_per_model=2*sum(map(len,(groups[s] for s in STAGES))),
             transmitted_model_bytes_per_model=2*sum(map(len,(groups[s] for s in STAGES)))*model_bytes),
         provenance='exposed-seed checkpoint feasibility; not an end-to-end defense result'))


def qualification(records):
    """Fixed exploratory advancement rule, evaluated only after all 72 results exist."""
    if len(records) != 72:
        return dict(status='pending',passed=False,results=len(records),required=72)
    index = {(r['seed'],r['condition'],r['checkpoint'],r['arm']):r for r in records}
    if len(index) != 72:
        raise ValueError('duplicate repair records')
    checks = []
    for seed in SEEDS:
        for rnd in (30,100):
            clean = index[seed,'none',rnd,'hard_patch']
            clean_base = index[seed,'none',rnd,'unchanged']
            proposed = index[seed,'model_replacement',rnd,'hard_patch']
            base = index[seed,'model_replacement',rnd,'unchanged']
            controls = [index[seed,'model_replacement',rnd,a] for a in ('clean_tune','random_patch')]
            asr = proposed['committed']['trigger_target_rate']
            control_gains = {r['arm']:r['raw']['trigger_target_rate']-asr for r in controls}
            values = dict(seed=seed,checkpoint=rnd,
                clean_accuracy_delta=clean['committed']['accuracy']-clean_base['committed']['accuracy'],
                attacked_accuracy_delta=proposed['committed']['accuracy']-base['committed']['accuracy'],
                attacked_accuracy_vs_clean=proposed['committed']['accuracy']-clean_base['committed']['accuracy'],
                asr=asr,asr_reduction=base['committed']['trigger_target_rate']-asr,
                gains_over_raw_equal_budget_controls=control_gains,
                clean_admitted=clean['decision']['passed'],attack_admitted=proposed['decision']['passed'])
            passed = (values['clean_admitted'] and values['attack_admitted']
                and min(values[k] for k in ('clean_accuracy_delta','attacked_accuracy_delta','attacked_accuracy_vs_clean')) >= -.02-1e-12
                and asr <= .20+1e-12 and values['asr_reduction'] >= .20-1e-12
                and all(g >= .10-1e-12 for g in control_gains.values()))
            checks.append(dict(values,passed=bool(passed)))
    passed = all(c['passed'] for c in checks)
    return dict(status='checkpoint_promising' if passed else 'checkpoint_rejected',passed=passed,checks=checks,
                rule='all mature seeds/checkpoints: utility within 2pp, ASR<=20%, reduction>=20pp, >=10pp ASR advantage over both raw equal-budget controls; both panels admit',
                early_checkpoint_role='round 11 is exploratory collapse analysis; excluded from mature feasibility gate',
                automatic_end_to_end=False,publication_ready=False,end_to_end_evidence=False)


def report():
    records=[]
    for row in manifest():
        if row['role'] != 'repair':
            continue
        folder=OUT/row['id']
        if (folder/'completion.json').exists():
            verify_seal(folder)
            records.extend(read(folder/'summary.json')['records'])
    verdict=qualification(records)
    dump(JOB/'qualification.json',verdict)
    dump(JOB/'results.json',dict(records=records,qualification=verdict))
    lines=['# V18 supervised checkpoint repair','',f"Status: {verdict['status']}; scored arms: {len(records)}/72.",
           'Exposed development data. Raw results precede admission; committed results include every hold. This is not continuous-poisoning or publication evidence.','',
           '| Condition | Seed | Round | Arm | Admitted | Raw accuracy | Raw trigger rate | Committed accuracy | Committed trigger rate |',
           '|---|---:|---:|---|---|---:|---:|---:|---:|']
    for r in records:
        lines.append(f"| {r['condition']} | {r['seed']} | {r['checkpoint']} | {r['arm']} | {r['decision']['passed']} | {100*r['raw']['accuracy']:.2f}% | {100*r['raw']['trigger_target_rate']:.2f}% | {100*r['committed']['accuracy']:.2f}% | {100*r['committed']['trigger_target_rate']:.2f}% |")
    (JOB/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return verdict


def execute(index):
    with exclusive_lock(JOB/'cell.lock'),threadpool_limits(limits=2):
        assert_frozen()
        row=manifest()[index]; folder=OUT/row['id']
        if folder.exists():
            raise FileExistsError('worker refuses an existing output directory')
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required for matched historical replay')
        if row['role']=='replay':
            run_replay(row,folder)
        else:
            run_repair(row,folder)
        dump(folder/'settings.json',dict(row=row,policy=asdict(POLICY),freeze_sha256=digest(JOB/'frozen-inputs.json')))
        assert_frozen()
        dump(folder/'completion.json',seal_files(folder))
        verify_seal(folder)


def run():
    with exclusive_lock(JOB/'coordinator.lock'):
        assert_frozen()
        state=dict(status='running',pid=os.getpid(),completed=0,total=24,failed=0,
                   started_utc=datetime.now(timezone.utc).isoformat(),publication_ready=False)
        dump(JOB/'status.json',state)
        dump(JOB/'environment.json',dict(torch=torch.__version__,cuda=torch.version.cuda,
             gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None))
        try:
            for index,row in enumerate(manifest()):
                assert_frozen()
                state.update(current=row['id'],phase=row['role'],updated_utc=datetime.now(timezone.utc).isoformat())
                dump(JOB/'status.json',state)
                folder=OUT/row['id']
                if (folder/'completion.json').exists():
                    verify_seal(folder)
                    if read(folder/'settings.json') != dict(row=row,policy=asdict(POLICY),freeze_sha256=digest(JOB/'frozen-inputs.json')):
                        raise ValueError('completed job settings changed')
                else:
                    if folder.exists():
                        if not folder.resolve().is_relative_to(OUT.resolve()):
                            raise ValueError('partial job outside output root')
                        folder.rename(folder.with_name(folder.name+f'.interrupted-{time.time_ns()}'))
                    print(f"START {index+1}/24 {row['id']}",flush=True)
                    child=subprocess.run([sys.executable,'-u','-m','scripts.trust_supervised_v18.study','--cell',str(index)],cwd=ROOT)
                    if child.returncode:
                        raise RuntimeError(f"worker failed with exit {child.returncode}: {row['id']}")
                    verify_seal(folder)
                state['completed']+=1
                if row['role']=='repair':
                    report()
                dump(JOB/'status.json',state)
                print(f"DONE {index+1}/24 {row['id']}",flush=True)
            verdict=report()
            state.update(status=verdict['status'],finished_utc=datetime.now(timezone.utc).isoformat())
            dump(JOB/'status.json',state)
        except Exception as error:
            state.update(status='failed',failed=1,error=str(error),traceback=traceback.format_exc(),
                         finished_utc=datetime.now(timezone.utc).isoformat())
            dump(JOB/'status.json',state)
            raise


def main():
    parser=argparse.ArgumentParser()
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--freeze',action='store_true')
    action.add_argument('--run',action='store_true')
    action.add_argument('--cell',type=int)
    action.add_argument('--report',action='store_true')
    args=parser.parse_args()
    if args.freeze:
        frozen=inputs(); path=JOB/'frozen-inputs.json'
        if path.exists() and read(path)!=frozen:
            raise ValueError('V18 already frozen with different inputs')
        dump(path,frozen)
        print(f'Frozen V18: {digest(path)}')
    elif args.run:
        run()
    elif args.report:
        report()
    else:
        execute(args.cell)


if __name__=='__main__':
    main()
