import copy
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from threadpoolctl import threadpool_limits

from scripts.trust_online_v21 import online,analysis,study
from scripts.trust_online_v21.build_runner import rendered_runner
from scripts.trust_online_v21.runner import BenchmarkRunner
from scripts.trust_alignment_v17.runner import BenchmarkRunner as OriginalRunner
from scripts.trust_repair_v16_fixed.integrity import snapshot,state_hash
from scripts.trust_supervised_v18.repair import RepairPolicy


@pytest.fixture(autouse=True)
def threads():
    with threadpool_limits(limits=2):
        torch.set_num_threads(2)
        yield


def factory():
    return nn.Sequential(nn.Conv2d(1,2,3),nn.ReLU(),nn.Flatten(),nn.Linear(72,2))


def inputs():
    rng=np.random.default_rng(32);x=rng.random((120,1,8,8),dtype=np.float32)
    y=np.tile([0,1],60).astype('int64')
    split=SimpleNamespace(validation=[np.arange(i*10,(i+1)*10) for i in range(12)])
    torch.manual_seed(5);model=factory()
    policy=replace(RepairPolicy(),repair_clients=2,panel_size=2,examples_per_client=4,
                   passes=1,steps_per_client=1,batch_size=2,proposals=2)
    return model,x,y,split,policy


def test_runner_is_exact_explicit_insertion_and_matrix_clean_first():
    assert (study.ROOT/'scripts/trust_online_v21/runner.py').read_text(encoding='utf-8')==rendered_runner()
    rows=study.manifest();assert len(rows)==36 and len({r['id'] for r in rows})==36
    assert all(r['attack']=='none' for r in rows[:12])
    assert all(r['attack']!='none' for r in rows[12:])
    controller=online.OnlineRepair('v20_repair')
    assert [r for r in range(102) if controller.due(r)]==list(range(10,101))
    with pytest.raises(ValueError,match='invalid fixed'):
        online.OnlineRepair('v20_repair',online.DeploymentPolicy(interval=0))


@pytest.mark.parametrize('arm',('v19_repair','v20_repair'))
def test_online_commit_hold_budget_and_disjoint_groups(monkeypatch,arm):
    model,x,y,split,p=inputs();original=state_hash(snapshot(model));rng=torch.get_rng_state().clone()
    # Exercise acceptance independently of the already-tested empirical verifier.
    monkeypatch.setattr(online,'verify',lambda before,after,policy:dict(
        passed=True,scored_model_hash=next(iter(after['screen'].values()))['model_hash']))
    controller=online.OnlineRepair(arm,repair=p)
    outcome=controller.attempt(model,x,y,split,10,[0,1],11)
    assert state_hash(snapshot(model))==original and torch.equal(rng,torch.get_rng_state())
    assert state_hash(snapshot(outcome.committed))==outcome.payload['candidate_model_hash']
    assert all(p.requires_grad for p in outcome.committed.parameters())
    assert outcome.payload['work']['optimizer_steps']==2
    assert outcome.payload['extra_transmitted_bytes']>outcome.payload['work']['transmitted_bytes']
    groups=outcome.partitions['groups'];members=[int(c) for g in groups.values() for c in g]
    assert len(members)==len(set(members))==6 and not set(members)&{0,1}
    assert outcome.payload['verification_presentations']==sum(m['example_evaluations'] for panel in outcome.panels.values()
        for group in panel.values() for m in group.values())
    monkeypatch.setattr(online,'verify',lambda before,after,policy:dict(
        passed=False,scored_model_hash=next(iter(after['screen'].values()))['model_hash']))
    rejected=controller.attempt(model,x,y,split,10,[0,1],11)
    assert rejected.payload['held'] and state_hash(snapshot(rejected.committed))==original
    assert rejected.telemetry==outcome.telemetry


def test_commit_rejects_scored_model_mismatch_and_insufficient_data():
    model,x,y,split,p=inputs()
    with pytest.raises(ValueError,match='after scoring'):
        online.select_committed(model,copy.deepcopy(model),dict(passed=True,scored_model_hash='bad'))
    controller=online.OnlineRepair('v20_repair',repair=p)
    with pytest.raises(ValueError,match='unscheduled'):controller.attempt(model,x,y,split,9,[0,1],11)
    tiny=SimpleNamespace(validation=[v[:1] for v in split.validation])
    with pytest.raises(ValueError,match='at least two'):controller.attempt(model,x,y,tiny,10,[0,1],11)


def rows():
    result=[]
    for t in analysis.ATTACKS:
        for s in analysis.SEEDS:
            for a in analysis.ARMS:
                targeted=t=='model_replacement'
                rate=.1 if a=='v20_repair' else .9
                result.append(dict(seed=s,attack=t,arm=a,final_accuracy=.8,
                    repair_attempts=91 if a in ('v19_repair','v20_repair') else 0,
                    repair_admissions=80 if a in ('v19_repair','v20_repair') else 0,
                    mean_post_onset_asr=rate if targeted else None,endpoint_asr=rate if targeted else None))
    return result


def test_clean_gate_blocks_attacks_and_final_gate_requires_every_seed():
    data=rows();assert analysis.clean_gate(data[:12])['passed']
    assert analysis.qualify(data)['passed']
    assert analysis.qualify(data[:8])['status']=='clean_screen_running'
    data[3]['final_accuracy']=.77
    assert analysis.qualify(data[:12])['status']=='clean_rejected'
    assert analysis.qualify(data[:12])['held_attack_cells']==24
    data=rows();data[-1]['mean_post_onset_asr']=.4
    assert not analysis.qualify(data)['passed']
    data=rows();data[15]['final_accuracy']=.77
    assert not analysis.qualify(data)['passed']


def tiny_config(name,rounds=3):
    c=study.config(11,'none','clip_only')
    return replace(c,experiment_id=name,num_clients=20,participation_rate=.2,num_classes=2,
        rounds=rounds,natural_splits_path=None,validation_fraction=.3,server_samples_per_class=2,
        batch_size=64,malicious_fraction=0.,device='cpu',attack_start_round=1)


def arrays():
    rng=np.random.default_rng(9)
    x=rng.random((800,1,8,8),dtype=np.float32);y=np.tile([0,1],400).astype('int64')
    tx=rng.random((80,1,8,8),dtype=np.float32);ty=np.tile([0,1],40).astype('int64')
    return x,y,tx,ty


def test_full_runner_without_repair_preserves_original_trajectory(tmp_path):
    x,y,tx,ty=arrays();c=tiny_config('parity')
    original=OriginalRunner(c,tmp_path/'old');original.challenge_arm='clip_only'
    new=BenchmarkRunner(c,tmp_path/'new');new.challenge_arm='clip_only'
    a=original.run_arrays(x,y,factory,(tx,ty));b=new.run_arrays(x,y,factory,(tx,ty))
    assert json.loads((tmp_path/'old/chain-trace.json').read_text())==json.loads((tmp_path/'new/chain-trace.json').read_text())
    for key in ('accuracy_history','participants','split_bindings','local_training_work','validation_work'):
        assert a[key]==b[key]
    assert state_hash(snapshot(new.final_model))==study.trace_hashes(tmp_path/'new')[-1]


def test_full_runner_commits_repair_and_next_round_uses_that_model(tmp_path):
    x,y,tx,ty=arrays();c=tiny_config('online')
    p=replace(RepairPolicy(),repair_clients=2,panel_size=2,examples_per_client=4,
              passes=1,steps_per_client=1,batch_size=2,proposals=2)
    runner=BenchmarkRunner(c,tmp_path/'online');runner.challenge_arm='clip_only'
    runner.online_controller=online.OnlineRepair('v20_repair',online.DeploymentPolicy(2,1,3),p)
    runner.offline_observer=study.OfflineObserver(tx,ty,0,[np.arange(len(ty))])
    runner.run_arrays(x,y,factory,(tx,ty))
    records=runner.offline_observer.records;assert [r['round'] for r in records]==[2,3]
    trace=study.trace_hashes(tmp_path/'online')
    assert all(r['committed']['model_hash']==trace[r['round']] for r in records)
    events=[json.loads(line) for line in (tmp_path/'online/events.jsonl').read_text().splitlines()]
    integrity=[e for e in events if e['event_type']=='model_integrity']
    assert len(integrity)==3 and all(e['payload']['expected_hash']==e['payload']['committed_hash'] for e in integrity)
    assert integrity[2]['payload']['parent_hash']==trace[2]
    assert all(p.requires_grad for p in runner.final_model.parameters())
    assert (tmp_path/'online/online/round-002/decision.json').exists()
