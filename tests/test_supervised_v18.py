import copy
from dataclasses import replace
import json

import numpy as np
import pytest
import torch
from torch import nn
from threadpoolctl import threadpool_limits

from scripts.trust_supervised_v18 import repair, study
from scripts.trust_supervised_v18.capture import capture
from scripts.trust_alignment_v17.runner import BenchmarkRunner
from scripts.trust_repair_v16_fixed.integrity import snapshot, state_hash


@pytest.fixture(autouse=True)
def threads():
    with threadpool_limits(limits=2):
        torch.set_num_threads(2)
        yield


def toy_model():
    torch.manual_seed(19)
    return nn.Sequential(nn.Flatten(),nn.Linear(64,2))


def toy_data():
    x=np.random.default_rng(9).random((32,1,8,8),dtype=np.float32)
    return x,(x[:,0,0,0]>.5).astype('int64')


def tiny_policy():
    return replace(repair.RepairPolicy(),repair_clients=2,panel_size=2,
                   passes=1,steps_per_client=2,batch_size=4,proposals=3)


def test_manifest_clean_first_finite_and_historical_versions_unchanged():
    rows=study.manifest()
    assert len(rows)==24 and len({r['id'] for r in rows})==24
    assert all(r['role']=='replay' for r in rows[:6])
    assert all(r['condition']=='none' for r in rows[6:15])
    assert {r['checkpoint'] for r in rows[6:]}=={11,30,100}
    assert study.replay_config(11,'none').vote_rule=='scaled_loss'
    assert study.replay_config(11,'none').rounds==100


def test_allocation_is_disjoint_reproducible_and_rejects_overlap():
    validation=[np.arange(i*10,(i+1)*10) for i in range(10)]
    a=repair.allocate(validation,[0,1],11,30,tiny_policy())
    b=repair.allocate(validation,[0,1],11,30,tiny_policy())
    assert repair.partition_binding(a)==repair.partition_binding(b)
    assert not {0,1}.intersection(c for members in a.values() for c in members)
    bad=copy.deepcopy(a)
    bad['verify'][next(iter(bad['verify']))]=next(iter(bad['repair'].values())).copy()
    with pytest.raises(ValueError,match='overlap'):
        repair.assert_disjoint(bad)
    with pytest.raises(ValueError,match='insufficient'):
        repair.allocate(validation,list(range(6)),11,30,tiny_policy())


def test_patch_preserves_parent_and_repair_arms_match_budget_and_rng():
    parent=toy_model(); before=state_hash(snapshot(parent))
    x,y=toy_data(); local={3:(x[:16],y[:16]),7:(x[16:],y[16:])}
    results={}
    rng=torch.get_rng_state().clone()
    for arm in repair.ARMS:
        candidate,telemetry=repair.repair_model(parent,local,arm,11,30,tiny_policy())
        assert state_hash(snapshot(parent))==before
        assert torch.equal(torch.get_rng_state(),rng)
        assert telemetry['model_hash']==state_hash(snapshot(candidate))
        results[arm]=(candidate,telemetry)
    assert results['unchanged'][1]['work']['optimizer_steps']==0
    budgets=[results[arm][1]['work'] for arm in repair.ARMS[1:]]
    assert all(b==budgets[0] for b in budgets)
    assert budgets[0]['optimizer_steps']==4
    assert budgets[0]['training_examples']==32
    again,t=repair.repair_model(parent,local,'hard_patch',11,30,tiny_policy())
    assert t==results['hard_patch'][1]
    assert state_hash(snapshot(again))==state_hash(snapshot(results['hard_patch'][0]))


def test_cpu_sgd_matches_independent_reference():
    model=toy_model(); original=snapshot(model)
    for parameter in model.parameters():
        parameter.grad=torch.ones_like(parameter)*.25
    expected={k:v-.1*.25 for k,v in original.items()}
    repair.cpu_sgd_step(model,.1)
    assert all(torch.equal(snapshot(model)[k],v) for k,v in expected.items())


def test_measurement_detects_constant_predictions_with_zero_patch_contrast():
    model=toy_model()
    with torch.no_grad():
        model[1].weight.zero_(); model[1].bias.copy_(torch.tensor([5.,0.]))
    x,y=toy_data(); result=repair.measure(model,x,y)
    assert result['constant_prediction']
    assert result['worst_patch_correct_count']==result['correct_count']
    panels={s:{0:result,1:result} for s in repair.STAGES}
    verdict=repair.verify(panels,panels,tiny_policy())
    assert not verdict['passed']
    assert all(v['constant_prediction'] for v in verdict['panels'].values())


def fake_measurement(hash_value='same',correct=8,loss=1.,patch=2.,counts=(5,5),labels=(5,5)):
    return dict(model_hash=hash_value,example_binding='identical-data',n=10,clean_losses=[loss]*10,worst_patch_losses=[patch]*10,
                correct_count=correct,prediction_counts=list(counts),label_counts=list(labels),
                label_class_count=sum(v>0 for v in labels),example_evaluations=1480)


def panels(value):
    return {s:{0:copy.deepcopy(value),1:copy.deepcopy(value)} for s in repair.STAGES}


def test_verification_requires_both_panels_correct_hash_and_utility():
    before=panels(fake_measurement())
    after=panels(fake_measurement(hash_value='candidate',patch=1.))
    assert repair.verify(before,after,tiny_policy())['passed']
    after['verify'][0]['model_hash']='different'
    with pytest.raises(ValueError,match='different models'):
        repair.verify(before,after,tiny_policy())
    after=panels(fake_measurement(correct=6,patch=1.))
    assert not repair.verify(before,after,tiny_policy())['passed']
    after['screen'][0]['example_binding']='different-data'
    with pytest.raises(ValueError,match='unpaired'):
        repair.verify(before,after,tiny_policy())
    after=panels(fake_measurement(patch=3.))
    assert not repair.verify(before,after,tiny_policy())['passed']
    after['verify'].pop(0)
    with pytest.raises(ValueError,match='incomplete'):
        repair.verify(before,after,tiny_policy())


def test_constant_guard_pools_label_diversity_across_clients():
    before=panels(fake_measurement(counts=(10,0),labels=(10,0)))
    for stage in repair.STAGES:
        before[stage][1]['label_counts']=[0,10]
    assert not repair.verify(before,before,tiny_policy())['passed']


def test_scored_committed_equality_and_failed_repair_retains_parent():
    parent=toy_model(); candidate=copy.deepcopy(parent)
    with torch.no_grad():
        candidate[1].bias.add_(1.)
    verdict=dict(passed=True,scored_model_hash=state_hash(snapshot(candidate)))
    committed=study.select_committed(parent,candidate,verdict)
    assert state_hash(snapshot(committed))==verdict['scored_model_hash']
    verdict['passed']=False
    committed=study.select_committed(parent,candidate,verdict)
    assert state_hash(snapshot(committed))==state_hash(snapshot(parent))
    verdict['scored_model_hash']='tampered'
    with pytest.raises(ValueError,match='changed'):
        study.select_committed(parent,candidate,verdict)


@pytest.mark.parametrize('condition',['none','model_replacement'])
def test_capture_does_not_change_any_training_model_hash(tmp_path,condition):
    c=replace(study.replay_config(11,condition),dataset='synthetic',num_classes=2,
              num_clients=20,participation_rate=.25,rounds=3,attack_start_round=2,
              natural_splits_path=None,device='cpu',server_samples_per_class=2,
              batch_size=16,dirichlet_alpha=100.)
    rng=np.random.default_rng(71)
    x=rng.random((1000,1,8,8),dtype=np.float32);y=(x[:,0,0,0]>.5).astype('int64')
    def factory():
        return nn.Sequential(nn.Flatten(),nn.Linear(64,2))
    # External arrays are kept disjoint in this synthetic replay parity check.
    tx=rng.random((100,1,8,8),dtype=np.float32);ty=(tx[:,0,0,0]>.5).astype('int64')
    base=BenchmarkRunner(c,tmp_path/'base')
    base.run_arrays(x,y,factory,(tx,ty))
    hashes=capture(c,tmp_path/'capture',x,y,factory,(tx,ty),(1,3))
    traces=[json.loads((tmp_path/name/'chain-trace.json').read_text()) for name in ('base','capture')]
    assert [r['model_hash'] for r in traces[0]]==[r['model_hash'] for r in traces[1]]
    for rnd,expected in hashes.items():
        checkpoint=torch.load(tmp_path/'capture'/'checkpoints'/f'round-{rnd:03}.pt',weights_only=True)
        assert state_hash(checkpoint['state'])==expected==traces[0][rnd]['model_hash']


def test_seal_tamper_and_checkpoint_tamper_stop(tmp_path):
    (tmp_path/'value.txt').write_text('original')
    study.dump(tmp_path/'completion.json',study.seal_files(tmp_path))
    study.verify_seal(tmp_path)
    (tmp_path/'value.txt').write_text('modified')
    with pytest.raises(ValueError,match='seal mismatch'):
        study.verify_seal(tmp_path)
    torch.save(dict(model_hash='wrong',state=snapshot(toy_model())),tmp_path/'bad.pt')
    with pytest.raises(ValueError,match='hash mismatch'):
        study.load_checkpoint(tmp_path/'bad.pt',toy_model)


def test_qualification_cannot_pass_incomplete_or_collapsed_results():
    assert study.qualification([])['status']=='pending'
    rows=[]
    for seed in study.SEEDS:
        for condition in study.CONDITIONS:
            for rnd in study.CHECKPOINTS:
                for arm in repair.ARMS:
                    rows.append(dict(seed=seed,condition=condition,checkpoint=rnd,arm=arm,
                        raw=dict(accuracy=.8,trigger_target_rate=.95),
                        committed=dict(accuracy=.8,trigger_target_rate=.95),decision=dict(passed=True)))
    assert not study.qualification(rows)['passed']
    for r in rows:
        if r['arm']=='hard_patch':
            r['raw']['trigger_target_rate']=.1;r['committed']['trigger_target_rate']=.1
    assert study.qualification(rows)['passed']
    rows[-1]['committed']['accuracy']=.05
    assert not study.qualification(rows)['passed']


def test_repair_api_has_no_evaluation_or_attack_configuration():
    import inspect
    parameters=set(inspect.signature(repair.repair_model).parameters)
    assert parameters=={'parent','local_data','arm','seed','checkpoint','policy'}
    assert set(inspect.signature(repair.verify).parameters)=={'parent_measurements','candidate_measurements','policy'}
