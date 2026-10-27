import copy
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn
from threadpoolctl import threadpool_limits

from scripts.trust_anchor_v20 import repair,study
from scripts.trust_interleave_v19 import repair as reference
from scripts.trust_supervised_v18.repair import RepairPolicy
from scripts.trust_repair_v16_fixed.integrity import snapshot,state_hash


@pytest.fixture(autouse=True)
def threads():
    with threadpool_limits(limits=2):
        torch.set_num_threads(2)
        yield


def inputs():
    rng=np.random.default_rng(13)
    x=rng.random((16,1,8,8),dtype=np.float32);y=(x[:,0,0,0]>.5).astype('int64')
    local={2:(x[:8],y[:8]),7:(x[8:],y[8:])}
    policy=replace(RepairPolicy(),repair_clients=2,passes=1,steps_per_client=2,batch_size=4,proposals=2)
    torch.manual_seed(41)
    model=nn.Sequential(nn.Conv2d(1,2,3),nn.ReLU(),nn.Flatten(),nn.Linear(72,2))
    return model,local,policy


def test_matrix_and_primary_are_predeclared():
    assert len(study.manifest())*len(repair.ARMS)==24
    assert len({r['id'] for r in study.manifest()})==6
    assert study.analyze([])['status']=='pending'
    assert repair.AnchorPolicy().suppression_weight==.1


def test_baseline_is_exact_every_optimizer_step():
    model,local,p=inputs()
    a,ta=reference.repair_model(model,local,'hard_patch','interleaved',11,100,p)
    b,tb=repair.repair_model(model,local,'hard_patch',11,100,p)
    assert ta==tb and state_hash(snapshot(a))==state_hash(snapshot(b))


@pytest.mark.parametrize('arm',repair.ARMS[1:])
def test_teacher_parent_convolutions_rng_and_budget_preserved(arm):
    model,local,p=inputs();before=snapshot(model);rng=torch.get_rng_state().clone()
    candidate,t=repair.repair_model(model,local,arm,11,100,p)
    assert state_hash(snapshot(model))==state_hash(before)==t['teacher_hash']
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(before[n],snapshot(candidate)[n]) for n in t['frozen_parameter_names'])
    assert not torch.equal(before['3.weight'],snapshot(candidate)['3.weight'])
    assert all(p.grad is None for p in model.parameters())
    assert t['work']['optimizer_steps']==4 and t['work']['teacher_examples']==16
    assert t['work']['model_transmissions']==8 and t['work']['teacher_transmissions']==2
    assert t['work']['proposal_examples']==32 and t['work']['training_examples']==32
    _,repeat=repair.repair_model(model,local,arm,11,100,p)
    assert t==repeat


def test_kl_direction_temperature_and_teacher_detachment():
    teacher=torch.tensor([[2.,-1.,.5]],requires_grad=True)
    clean=torch.tensor([[1.,-.3,.1]],requires_grad=True)
    patch=torch.tensor([[.2,2.,-.1]],requires_grad=True);y=torch.tensor([0])
    loss,terms=repair.loss_terms(clean,patch,teacher,y,'anchored_patch')
    q=torch.softmax(teacher.detach()/2,1)
    expected=4*(q*(q.log()-torch.log_softmax(patch/2,1))).sum()
    assert torch.allclose(terms['patch_kl'],expected,atol=1e-6)
    assert torch.allclose(loss,.5*terms['ce']+2*terms['clean_kl']+2*expected)
    loss.backward();assert teacher.grad is None
    assert clean.grad is not None and patch.grad is not None


def test_no_teacher_self_target_or_patch_gradient_in_clean_arm():
    teacher=torch.tensor([[2.,-1.]])
    clean=teacher.clone().requires_grad_(True);patch=torch.tensor([[-2.,1.]],requires_grad=True)
    loss,terms=repair.loss_terms(clean,patch,teacher,torch.tensor([0]),'anchored_clean')
    assert abs(float(terms['clean_kl'].detach()))<1e-6
    loss.backward();assert patch.grad is None and clean.grad.abs().sum()>0


def test_suppression_excludes_label_and_exposes_offset_sensitivity():
    teacher=torch.tensor([[1.,0.,0.]])
    clean=teacher.clone();patch=torch.tensor([[10.,2.,-2.]])
    _,parts=repair.loss_terms(clean,patch,teacher,torch.tensor([0]),'anchored_suppress')
    assert float(parts['suppression'])==4.
    _,shift=repair.loss_terms(clean,patch+5,teacher,torch.tensor([0]),'anchored_suppress')
    assert float(shift['suppression'])==58.
    assert torch.allclose(parts['patch_kl'],shift['patch_kl'],atol=1e-5)


def test_invalid_plan_and_unsupported_network_rejected():
    model,local,p=inputs();plan=reference.make_plan(local,11,100,p);plan[0]['ids'][0]=7
    with pytest.raises(ValueError,match='plan differs'):
        repair.repair_model(model,local,'anchored_patch',11,100,p,plan)
    with pytest.raises(ValueError,match='convolutional'):
        repair.freeze_convolutions(nn.Linear(2,2))
    with pytest.raises(ValueError,match='stateless'):
        repair.repair_model(nn.Sequential(model,nn.Dropout(.2)),local,'anchored_clean',11,100,p)


def synthetic_records():
    rows=[]
    for s in study.SEEDS:
        for t in study.CONDITIONS:
            for a in repair.ARMS:
                rate=.1 if a=='anchored_suppress' else .6
                rows.append(dict(seed=s,condition=t,arm=a,parent_model_hash='p',plan_sha256='plan',
                    decision=dict(passed=True),raw=dict(accuracy=.8,trigger_target_rate=rate),
                    committed=dict(accuracy=.8,trigger_target_rate=rate),
                    baseline=dict(accuracy=.8,trigger_target_rate=.99)))
    return rows


def test_qualification_uses_all_seeds_committed_outcomes_and_rejects_mismatch():
    rows=synthetic_records();assert study.analyze(rows)['passed']
    rows[-1]['decision']['passed']=False;assert not study.analyze(rows)['passed']
    rows=synthetic_records();rows[-1]['committed']['trigger_target_rate']=.99
    assert not study.analyze(rows)['passed']
    rows=synthetic_records();rows[3]['committed']['accuracy']=.77
    assert not study.analyze(rows)['passed']
    rows=synthetic_records();rows[0]['plan_sha256']='wrong'
    with pytest.raises(ValueError,match='paired inputs'):study.analyze(rows)
    rows=synthetic_records();rows[-1]=copy.deepcopy(rows[0])
    with pytest.raises(ValueError,match='duplicate'):study.analyze(rows)
