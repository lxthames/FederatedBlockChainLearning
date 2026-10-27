from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from threadpoolctl import threadpool_limits

from scripts.trust_periodic_v22 import online,analysis,study
from scripts.trust_periodic_v22.build_sources import runner_source,analysis_source,study_source
from scripts.trust_online_v21.online import OnlineRepair as BaseRepair
from scripts.trust_repair_v16_fixed.integrity import snapshot,state_hash
from scripts.trust_supervised_v18.repair import RepairPolicy


@pytest.fixture(autouse=True)
def threads():
    with threadpool_limits(limits=2):
        torch.set_num_threads(2)
        yield


def factory():return nn.Sequential(nn.Conv2d(1,2,3),nn.ReLU(),nn.Flatten(),nn.Linear(72,2))


def inputs():
    rng=np.random.default_rng(15);x=rng.random((120,1,8,8),dtype=np.float32)
    y=np.tile([0,1],60).astype('int64');torch.manual_seed(9)
    split=SimpleNamespace(validation=[np.arange(10*i,10*(i+1)) for i in range(12)])
    policy=replace(RepairPolicy(),repair_clients=2,panel_size=2,examples_per_client=4,
        passes=1,steps_per_client=1,batch_size=2,proposals=2)
    return factory(),x,y,split,policy


def test_sources_and_matrix_are_frozen_explicit_derivations():
    for name,render in (('runner.py',runner_source),('analysis.py',analysis_source),('study.py',study_source)):
        assert (study.ROOT/'scripts/trust_periodic_v22'/name).read_text(encoding='utf-8')==render()
    rows=study.manifest();assert len(rows)==36 and len({r['id'] for r in rows})==36
    assert all(r['attack']=='none' for r in rows[:12])
    c=online.OnlineRepair('v20_peak')
    assert [r for r in range(101) if c.due(r)]==[25,50,75,100]
    assert online.MemoryPolicy().step_tolerance==.0025 and online.MemoryPolicy().peak_tolerance==.015


def test_gate_uses_both_step_and_cumulative_floor_on_paired_examples():
    def count(n):return {1:dict(n=1000,correct=n,binding='same')}
    assert online.panel_decision(count(800),count(798),{1:800})['passed']
    assert not online.panel_decision(count(800),count(797),{1:800})['passed']
    assert not online.panel_decision(count(790),count(790),{1:820})['passed']
    assert online.panel_decision(count(790),count(805),{1:820})['passed']
    with pytest.raises(ValueError,match='incomplete'):online.panel_decision(count(1),{}, {1:2})
    with pytest.raises(ValueError,match='examples'):online.panel_decision(count(800),count(800),{1:700})
    altered=count(800);altered[1]['binding']='different'
    with pytest.raises(ValueError,match='examples'):online.panel_decision(count(800),altered,{1:800})


def test_running_peaks_fixed_bindings_rng_and_missing_rounds():
    model,x,y,split,p=inputs();controller=online.OnlineRepair('v20_peak',repair=p)
    original=state_hash(snapshot(model));rng=torch.get_rng_state().clone()
    a=controller.observe(model,x,y,split,1,11)
    previous=controller.peaks.copy();ids={c:v.copy() for c,v in controller.ids.items()}
    with torch.no_grad():model[-1].weight.mul_(0);model[-1].bias.copy_(torch.tensor([10.,-10.]))
    controller.observe(model,x,y,split,2,11)
    assert all(controller.peaks[c]>=previous[c] for c in previous)
    assert all(np.array_equal(ids[c],controller.ids[c]) for c in ids)
    assert torch.equal(rng,torch.get_rng_state())
    assert a['model_hash']==original and a['example_presentations']==48
    with pytest.raises(ValueError,match='every round'):controller.observe(model,x,y,split,4,11)
    with pytest.raises(ValueError,match='current private reference'):
        controller.attempt(model,x,y,split,25,[0,1],11)


def test_inner_loop_is_identical_and_extra_gate_can_only_remove_admission(monkeypatch):
    model,x,y,split,p=inputs();deployment=online.DeploymentPolicy(1,1,2)
    baseline=BaseRepair('v20_repair',deployment,p).attempt(model,x,y,split,1,[0,1],11)
    results=[]
    for variant in ('v20_periodic','v20_peak'):
        c=online.OnlineRepair(variant,deployment,p);c.observe(model,x,y,split,1,11)
        # Isolate memory veto behavior from the unmodified empirical verifier.
        monkeypatch.setattr(online,'panel_decision',lambda *args:dict(passed=False))
        out=c.attempt(model,x,y,split,1,[0,1],11)
        assert out.telemetry==baseline.telemetry
        assert out.payload['candidate_model_hash']==baseline.payload['candidate_model_hash']
        assert out.payload['decision']['base_verifier_passed']==baseline.payload['decision']['passed']
        if variant=='v20_peak':
            assert not out.payload['decision']['passed']
            assert state_hash(snapshot(out.committed))==state_hash(snapshot(model))
        else:assert out.payload['decision']['passed']==baseline.payload['decision']['passed']
        results.append(out)
    assert results[0].payload['memory_candidate_presentations']==results[1].payload['memory_candidate_presentations']


def test_clean_gate_requires_real_admission_and_each_seed():
    rows=[dict(seed=s,attack='none',arm=a,final_accuracy=.8,repair_attempts=4,
               repair_admissions=1) for s in analysis.SEEDS for a in analysis.ARMS]
    assert analysis.clean_gate(rows)['passed']
    rows[3]['repair_admissions']=0;assert not analysis.clean_gate(rows)['passed']
    rows[3]['repair_admissions']=1;rows[3]['final_accuracy']=.77
    assert analysis.qualify(rows)['status']=='clean_rejected'


def test_full_runner_reference_and_commit_integrity(tmp_path):
    rng=np.random.default_rng(4);x=rng.random((800,1,8,8),dtype=np.float32)
    y=np.tile([0,1],400).astype('int64');tx=x[:80].copy();ty=y[:80].copy()
    # Software integration fixture only; real studies use disjoint natural test data.
    c=replace(study.config(11,'none','clip_only'),experiment_id='v22-toy',num_clients=20,
        participation_rate=.2,num_classes=2,rounds=3,natural_splits_path=None,
        validation_fraction=.3,server_samples_per_class=2,batch_size=64,malicious_fraction=0.,
        device='cpu',attack_start_round=1)
    p=replace(RepairPolicy(),repair_clients=2,panel_size=2,examples_per_client=4,
        passes=1,steps_per_client=1,batch_size=2,proposals=2)
    runner=study.BenchmarkRunner(c,tmp_path/'run');runner.challenge_arm='clip_only'
    runner.online_controller=online.OnlineRepair('v20_peak',online.DeploymentPolicy(2,1,3),p)
    runner.offline_observer=study.OfflineObserver(tx,ty,0,[np.arange(80)])
    runner.run_arrays(x,y,factory,(tx,ty))
    events=[json.loads(line) for line in (tmp_path/'run/events.jsonl').read_text().splitlines()]
    monitors=[e for e in events if e['event_type']=='private_reference_monitor']
    assert [e['round_index'] for e in monitors]==[1,2,3]
    trace=study.trace_hashes(tmp_path/'run')
    for r in runner.offline_observer.records:assert r['committed']['model_hash']==trace[r['round']]
    assert all(p.requires_grad for p in runner.final_model.parameters())
