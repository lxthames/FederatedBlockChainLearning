import copy
from dataclasses import replace
import json
import numpy as np
import pytest
import torch
from torch import nn
from threadpoolctl import threadpool_limits
from scripts.trust_literature_v6.quasi_newton import SecantHessian
from scripts.trust_literature_v6.literature import (cosine_distances,flame,filter_reports,
    PartialFLDetector,FLShield,gap_two_clusters)
from scripts.trust_literature_v6.research.contracts import ClientUpdate
from scripts.trust_literature_v6.research.config import ExperimentConfig
from scripts.trust_literature_v6.research.runner_v2 import BenchmarkRunner
from scripts.trust_literature_v6.research.splits import make_splits
from scripts.trust_literature_v6.recovery import RecoveryPilot,detect_archive,selective_plan,recovery_binding
from scripts.trust_literature_v6.study import manifest,clean_gate


@pytest.fixture(autouse=True)
def bounded_threads():
    with threadpool_limits(limits=2):yield


def test_hessian_product_matches_dense_bfgs_and_secant():
    rng=np.random.default_rng(9);matrix=np.diag([1.,2.,4.,8.]);h=SecantHessian(3)
    pairs=[]
    for _ in range(3):
        s=rng.normal(size=4);y=matrix@s;assert h.add(s,y);pairs.append((s,y))
    s,y=pairs[-1];dense=np.eye(4)*(s@y)/(s@s)
    for s,y in pairs:
        bs=dense@s;dense=dense-np.outer(bs,bs)/(s@bs)+np.outer(y,y)/(s@y)
    v=rng.normal(size=4)
    np.testing.assert_allclose(h.product(v),dense@v,rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(h.product(pairs[-1][0]),pairs[-1][1],rtol=1e-10,atol=1e-10)
    assert not h.add([1,2],[-1,-2])
    with pytest.raises(ValueError):h.add([1,2],[1])


def test_cosine_zero_vectors_and_majority_flame_noise():
    distance=cosine_distances([[0,0],[0,0],[1,0],[-1,0]])
    assert np.isfinite(distance).all() and distance[0,1]==0 and distance[2,3]==2
    model=nn.Linear(2,1,bias=False)
    with torch.no_grad():model.weight.zero_()
    vectors=[[1,.01],[1,.02],[1,0],[1,-.01],[-10,4],[-10,5],[-10,6]]
    updates=[ClientUpdate(i,{'weight':torch.tensor([v],dtype=torch.float32)},1) for i,v in enumerate(vectors)]
    a=flame(model,updates,11,1,0);b=flame(model,updates,11,1,0)
    assert len(a.accepted_client_ids)>=4
    torch.testing.assert_close(a.state['weight'],b.state['weight'])
    assert float(a.state['weight'].norm())<=a.diagnostics['clip_bound']+1e-6
    noisy=flame(model,updates,11,1,.001)
    assert not torch.equal(a.state['weight'],noisy.state['weight'])
    torch.testing.assert_close(noisy.state['weight'],flame(model,updates,11,1,.001).state['weight'])


def test_report_filter_requires_observed_classes_and_keeps_equality():
    report=np.ones((12,2,3));report[:3,0,0]=np.nan
    scores,keep,_=filter_reports(report,11)
    np.testing.assert_allclose(scores,[1,1]);assert keep.all()
    report[:,:,2]=np.nan
    with pytest.raises(ValueError,match='class'):filter_reports(report,11)


def cfg(**changes):
    return ExperimentConfig(**(dict(experiment_id='v6-test',dataset='synthetic',num_clients=6,
        participation_rate=1,rounds=8,seed=11,device='cpu',server_samples_per_class=2,
        defense='fedavg',batch_size=32,learning_rate=.05,archive_updates=True)|changes))


def arrays():
    rng=np.random.default_rng(5);y=np.tile([0,1],150)
    return (rng.normal(size=(300,2))+y[:,None]).astype(np.float32),y


def test_shield_sparse_classes_fail_instead_of_fabricating_validation(tmp_path):
    x,y=arrays();c=cfg(defense='flshield_cluster',shield_min_class_samples=100,shield_max_class_samples=200)
    runner=BenchmarkRunner(c,tmp_path);split=make_splits(y,c,external_test=True)
    with pytest.raises(ValueError,match='coverage'):FLShield(runner,x,y,split)
    with pytest.raises(ValueError,match='malicious report'):cfg(defense='flshield_cluster',malicious_validator_fraction=.3)


def test_detector_decisions_are_causal_and_need_histories():
    model=nn.Linear(2,1,bias=False);first=PartialFLDetector(11);second=PartialFLDetector(11)
    for rnd in range(1,9):
        with torch.no_grad():model.weight.fill_(rnd*.01)
        updates=[ClientUpdate(i,{'weight':torch.tensor([[i*.001+rnd*.0001,.001]])},1) for i in range(8) if (i+rnd)%3]
        assert first.observe(model,updates,rnd)==second.observe(model,copy.deepcopy(updates),rnd)
    # No future response enters the prefix histories or decisions.
    before=copy.deepcopy(dict(first.strikes))
    second.observe(model,[ClientUpdate(i,{'weight':torch.ones(1,2)*1000},1) for i in range(8)],9)
    assert dict(first.strikes)==before
    flags,report=gap_two_clusters([1]*10,np.random.default_rng(11))
    assert not flags.any() and report['clusters']==1
    flags,report=gap_two_clusters([.01]*10+[.5]*10,np.random.default_rng(11))
    assert report['clusters']==2 and flags.tolist()==[False]*10+[True]*10


@pytest.mark.parametrize('defense',['flame','flshield_cluster','fldetector_pp'])
def test_new_defenses_run_without_heldout_label_influence(tmp_path,defense):
    x,y=arrays();ex=x[:24]+.01;ey=y[:24]
    c=cfg(defense=defense,rounds=2,shield_min_class_samples=0,shield_max_class_samples=10,archive_updates=False)
    outputs=[]
    for name,labels in [('a',ey),('b',1-ey)]:
        s=BenchmarkRunner(c,tmp_path/name).run_arrays(x,y,lambda:nn.Linear(2,2),(ex,labels))
        assert s['status']=='completed'
        outputs.append(json.loads((tmp_path/name/'chain-trace.json').read_text()))
    assert outputs[0]==outputs[1]


def test_conditional_recovery_preserves_source_and_accounts_requests(tmp_path):
    x,y=arrays();ex=x[:24]+.01;ey=y[:24];c=cfg()
    source=tmp_path/'source'
    baseline=BenchmarkRunner(c,source).run_arrays(x,y,lambda:nn.Linear(2,2),(ex,ey))
    trace=(source/'chain-trace.json').read_bytes()
    suspects=[0]
    reports=[]
    for method in ('retrain','fedrecover_partial','crab_fixedwindow'):
        path=tmp_path/method
        r=RecoveryPilot(replace(c,experiment_id=f'v6-{method}'),path).recover(
            source,method,suspects,'oracle_diagnostic',x,y,ex,ey,lambda:nn.Linear(2,2))
        assert r['status']=='completed' and r['identity_role']=='oracle_diagnostic'
        assert r['honest_clients_quarantined']==1
        assert not r['publication_ready'] and r['extra_transmitted_bytes']>=0
        steps=json.loads((path/'recovery-trace.json').read_text())[1:]
        assert all(0 not in r['participants'] for r in steps)
        assert r['exact_client_requests']==sum(len(s['exact_clients']) for s in steps)
        reports.append(r)
    assert reports[0]['exact_client_requests']==5*8
    assert reports[0]['estimated_client_requests']==0
    assert (source/'chain-trace.json').read_bytes()==trace
    unsupported=RecoveryPilot(c,tmp_path/'none').recover(source,'retrain',[],
        'detector_selected',x,y,ex,ey,lambda:nn.Linear(2,2))
    assert unsupported['status']=='unsupported_no_detected_suspects'
    detected,_=detect_archive(source,lambda:nn.Linear(2,2),11,8)
    assert isinstance(detected,list)
    before=recovery_binding(source,c,'retrain',[0],'detector_selected')
    assert before!=recovery_binding(source,c,'retrain',[1],'detector_selected')
    assert before!=recovery_binding(source,c,'retrain',[0],'oracle_diagnostic')
    (source/'chain-trace.json').write_bytes(trace+b'\n')
    assert before!=recovery_binding(source,c,'retrain',[0],'detector_selected')


def test_fedrecover_exercises_estimated_update_path(tmp_path):
    x,y=arrays();ex=x[:24]+.01;ey=y[:24];c=cfg(rounds=18)
    source=tmp_path/'source'
    BenchmarkRunner(c,source).run_arrays(x,y,lambda:nn.Linear(2,2),(ex,ey))
    report=RecoveryPilot(c,tmp_path/'recovery').recover(source,'fedrecover_partial',[0],
        'oracle_diagnostic',x,y,ex,ey,lambda:nn.Linear(2,2))
    assert report['estimated_client_requests']>0
    assert report['exact_client_requests']+report['estimated_client_requests']==18*5


def test_manifest_budget_and_incomplete_clean_gate():
    rows=manifest();assert len(rows)==18
    assert sum(r['config']['attack']=='none' for r in rows)==6
    assert all(r['config']['seed']==11 and r['config']['attack_start_round']==11 for r in rows)
    assert not clean_gate([],'flame')['passed']
