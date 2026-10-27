import copy
from dataclasses import replace
import json

import numpy as np
import pytest
import torch
from torch import nn
from threadpoolctl import threadpool_limits

from scripts.trust_interleave_v19 import repair
from scripts.trust_interleave_v19_fixed import study
from scripts.trust_supervised_v18 import repair as reference
from scripts.trust_repair_v16_fixed.integrity import snapshot,state_hash
from scripts.trust_repair_v16_fixed.research.splits import stream


@pytest.fixture(autouse=True)
def threads():
    with threadpool_limits(limits=2):
        torch.set_num_threads(2)
        yield


def inputs():
    rng=np.random.default_rng(13)
    x=rng.random((24,1,8,8),dtype=np.float32);y=(x[:,0,0,0]>.5).astype('int64')
    local={2:(x[:12],y[:12]),7:(x[12:],y[12:])}
    policy=replace(reference.RepairPolicy(),repair_clients=2,panel_size=2,
                   passes=2,steps_per_client=3,batch_size=4,proposals=3)
    torch.manual_seed(31)
    model=nn.Sequential(nn.Flatten(),nn.Linear(64,2))
    return model,local,policy


def test_frozen_matrix_is_true_2x2_and_baselines_all_precede_treatments():
    rows=study.manifest()
    assert len(rows)==12 and len({r['id'] for r in rows})==12
    assert all(r['schedule']=='sequential' for r in rows[:6])
    assert all(r['schedule']=='interleaved' for r in rows[6:])
    assert all(r['condition']=='none' for r in rows[:3]+rows[6:9])
    assert all(r['checkpoint']==100 for r in rows)
    assert len(rows)*len(repair.OBJECTIVES)==24


def test_precomputed_plan_preserves_v18_draws_and_reorders_same_updates():
    _,local,p=inputs();plan=repair.make_plan(local,11,100,p)
    rng=stream(11,'supervised-v18-repair',100,0,2)
    assert plan[0]['rng_seed']==int(rng.integers(2**31))
    assert plan[0]['ids']==rng.choice(12,4,replace=False).tolist()
    assert plan[0]['choices']==rng.integers(len(repair.transformations(8,8)),size=(3,4)).tolist()
    seq=repair.scheduled(plan,'sequential');mixed=repair.scheduled(plan,'interleaved')
    assert [s['client'] for s in seq[:6]]==[2,2,2,7,7,7]
    assert [s['client'] for s in mixed[:6]]==[2,7,2,7,2,7]
    assert sorted(seq,key=lambda s:(s['epoch'],s['client'],s['step']))==sorted(mixed,key=lambda s:(s['epoch'],s['client'],s['step']))
    with pytest.raises(ValueError,match='duplicate'):
        repair.scheduled(plan+[plan[0]],'interleaved')


@pytest.mark.parametrize('objective',repair.OBJECTIVES)
def test_sequential_matches_independent_v18_every_step_and_work(objective):
    model,local,p=inputs()
    expected,old=reference.repair_model(model,local,objective,11,100,p)
    candidate,new=repair.repair_model(model,local,objective,'sequential',11,100,p)
    assert new['steps']==old['steps']
    assert new['model_hash']==old['model_hash']==state_hash(snapshot(expected))
    assert new['model_hash']==state_hash(snapshot(candidate))
    assert {k:new['work'][k] for k in old['work']}==old['work']


def test_interleaving_preserves_rng_parent_plan_and_work_and_counts_transfers():
    model,local,p=inputs();before=state_hash(snapshot(model));rng=torch.get_rng_state().clone()
    records={}
    for schedule in repair.SCHEDULES:
        for objective in repair.OBJECTIVES:
            candidate,t=repair.repair_model(model,local,objective,schedule,11,100,p)
            assert torch.equal(torch.get_rng_state(),rng)
            assert state_hash(snapshot(model))==before
            assert state_hash(snapshot(candidate))==t['model_hash']
            records[schedule,objective]=t
    for key in ('optimizer_steps','training_examples','proposal_examples','diagnostic_examples'):
        assert len({t['work'][key] for t in records.values()})==1
    assert len({t['plan_sha256'] for t in records.values()})==1
    seq=records['sequential','clean_tune'];mixed=records['interleaved','clean_tune']
    assert seq['work']['model_transmissions']==8
    assert mixed['work']['model_transmissions']==24
    assert [o['after_update'] for o in seq['observations']]==[0,3,6,9,12]
    assert mixed['model_hash']!=seq['model_hash']
    _,repeat=repair.repair_model(model,local,'hard_patch','interleaved',11,100,p)
    assert repeat==records['interleaved','hard_patch']


def test_plan_tampering_and_stochastic_architecture_are_rejected():
    model,local,p=inputs();plan=repair.make_plan(local,11,100,p)
    plan[0]['ids'][0]=(plan[0]['ids'][0]+1)%12
    with pytest.raises(ValueError,match='plan differs'):
        repair.repair_model(model,local,'hard_patch','interleaved',11,100,p,plan)
    model.add_module('dropout',nn.Dropout())
    with pytest.raises(ValueError,match='stateless'):
        repair.repair_model(model,local,'hard_patch','interleaved',11,100,p)


def test_baseline_parity_rejects_step_change(tmp_path):
    _,local,p=inputs();model,_,_=inputs()
    _,old=reference.repair_model(model,local,'clean_tune',11,100,p)
    record=dict(objective='clean_tune',raw={},committed={},decision={},held=False,parent_model_hash='parent')
    study.dump(tmp_path/'clean_tune-repair.json',old)
    study.dump(tmp_path/'clean_tune-result.json',record)
    study.dump(tmp_path/'completion.json',{})
    new=copy.deepcopy(old)
    assert study.baseline_parity(new,record,tmp_path)['passed']
    new['steps'][0]['model_hash']='altered'
    with pytest.raises(ValueError,match='steps'):
        study.baseline_parity(new,record,tmp_path)


def synthetic_results():
    rows=[]
    for seed in study.SEEDS:
        for condition in study.CONDITIONS:
            for schedule in repair.SCHEDULES:
                for objective in repair.OBJECTIVES:
                    accuracy=.8 if schedule=='interleaved' else .75
                    rate=.1 if schedule=='interleaved' and objective=='hard_patch' else .95
                    rows.append(dict(seed=seed,condition=condition,schedule=schedule,objective=objective,
                        parent_model_hash=f'{seed}-{condition}',plan_sha256=str(seed),
                        raw=dict(accuracy=accuracy,trigger_target_rate=rate),committed=dict(accuracy=accuracy,trigger_target_rate=rate),
                        baseline=dict(accuracy=.8,trigger_target_rate=.99),decision=dict(passed=True),
                        work=dict(optimizer_steps=200,training_examples=6400,proposal_examples=12800,diagnostic_examples=1760)))
    return rows


def test_analysis_requires_complete_paired_plan_budgets_and_all_seed_targets(monkeypatch):
    monkeypatch.setattr(study,'read',lambda path:dict(raw=dict(trigger_target_rate=.95)))
    monkeypatch.setattr(study.parent,'verify_seal',lambda path:None)
    assert study.analyze([])['status']=='pending'
    rows=synthetic_results();result=study.analyze(rows)
    assert result['passed'] and len(result['paired_effects'])==12 and len(result['interactions'])==6
    assert result['paired_effects'][0]['raw_accuracy_gain']==pytest.approx(.05)
    broken=copy.deepcopy(rows);broken[-1]['plan_sha256']='other'
    with pytest.raises(ValueError,match='inputs differ'):
        study.analyze(broken)
    broken=copy.deepcopy(rows);broken[-1]['work']['optimizer_steps']=100
    with pytest.raises(ValueError,match='work differs'):
        study.analyze(broken)
    broken=copy.deepcopy(rows);broken[-1]['committed']['trigger_target_rate']=.5
    assert not study.analyze(broken)['passed']


def test_interleaving_cannot_start_without_every_baseline(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'OUT',tmp_path)
    with pytest.raises(FileNotFoundError):
        study.require_baselines()


def test_json_roundtrip_restores_original_panel_reduction_order():
    original={'screen':{40:{'values':[.3,.8]},5:{'values':[.1,.9]}},
              'verify':{27:{'values':[.2,.4]},8:{'values':[.6,.5]}}}
    groups={stage:{client:np.array([client]) for client in members} for stage,members in original.items()}
    serialized=json.loads(json.dumps(original,sort_keys=True))
    assert list(serialized['screen'])==['5','40']
    restored=study.restore_panel_order(serialized,groups)
    assert list(restored['screen'])==[40,5]
    assert list(restored['verify'])==[27,8]
    assert restored==original
    for stage in original:
        assert [v for m in restored[stage].values() for v in m['values']]==[v for m in original[stage].values() for v in m['values']]
    del serialized['screen']['5']
    with pytest.raises(ValueError,match='members differ'):
        study.restore_panel_order(serialized,groups)
