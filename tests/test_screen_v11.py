from dataclasses import replace
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from threadpoolctl import threadpool_limits

from scripts.trust_screen_v11 import study
from scripts.trust_screen_v11.loss_voting import LossCommittee,paired_decision
from scripts.trust_screen_v11.research.config import ExperimentConfig
from scripts.trust_screen_v11.research.contracts import ClientUpdate
from scripts.trust_screen_v11.research.runner_v2 import BenchmarkRunner
from scripts.trust_screen_v11.research.runner_helpers import _apply_delta
from scripts.trust_screen_v11.research.validation import VoteAggregator
from scripts.trust_screen_v11.pairing import compare_pair
from scripts.trust_mechanism_v10.research.config import ExperimentConfig as V10Config
from scripts.trust_mechanism_v10.research.runner_v2 import BenchmarkRunner as V10Runner


@pytest.fixture(autouse=True)
def threads():
    with threadpool_limits(limits=2):
        torch.set_num_threads(2)
        yield


def small(arm='scaled_quorum',attack='none'):
    return replace(study.config('fashion_mnist',arm,11,attack),dataset='synthetic',num_classes=2,
        num_clients=5,participation_rate=1.,rounds=3,attack_start_round=2,
        natural_splits_path=None,device='cpu',server_samples_per_class=2,
        batch_size=16,dirichlet_alpha=.5)


def arrays():
    x=np.random.default_rng(71).normal(size=(200,3)).astype('float32')
    return x,(x[:,0]>0).astype('int64')


def runner_stub(c):
    return SimpleNamespace(config=c,device=torch.device('cpu'),validation_work={
        'private_example_evaluations':0,'private_forward_passes':0,
        'reference_example_evaluations':0,'reference_forward_passes':0})


def test_manifest_clean_first_finite_and_no_original_attack():
    rows=study.manifest()
    assert len(rows)==96 and len({r['config']['experiment_id'] for r in rows})==96
    assert all(r['config']['attack']=='none' for r in rows[:36])
    assert all(r['config']['attack']!='none' for r in rows[36:])
    assert not any(r['arm']=='rank_original' for r in rows[36:])
    c=small()
    assert c.loss_tolerance==0 and c.loss_standard_errors==2
    assert c.trust_reward==c.rejection_trust_penalty==0
    for field in ('trust_updates','trust_weighting','trust_eligibility'):
        with pytest.raises(ValueError):replace(c,**{field:'dynamic'})


def test_loss_rule_has_no_forced_rejection_and_can_reject_all():
    c=small();model=nn.Linear(1,2,bias=False)
    updates=[ClientUpdate(i,{'weight':torch.ones(2,1)*.01},1) for i in range(5)]
    committee=LossCommittee(runner_stub(c),model,np.ones((4,1)),np.array([0,1,0,1]),
        SimpleNamespace(validation=[np.arange(4)]*6),updates)
    committee.pair=lambda validator,client:(np.ones(4),np.ones(4)-.01*(client+1))
    for client in range(5):
        v,d=committee.vote(5,client,False)
        assert v.approve and d['relative_rank_rejected'] is None
        assert not d['confidence_guarantee'] and not d['establishes_misconduct']
    committee.pair=lambda validator,client:(np.ones(4),np.ones(4)+.01*(client+1))
    assert not any(committee.vote(5,i,False)[0].approve for i in range(5))


def test_singleton_is_abstention_not_negative_vote():
    c=small();updates=[ClientUpdate(0,{'weight':torch.ones(2,1)*.01},1)]
    committee=LossCommittee(runner_stub(c),nn.Linear(1,2,bias=False),np.ones((1,1)),np.array([0]),
        SimpleNamespace(validation=[np.arange(1)]*5),updates)
    committee.pair=lambda validator,client:(np.ones(1),np.ones(1)*.5)
    vote,diag=committee.vote(2,0,False)
    assert diag['abstain'] and not vote.approve
    _,diag=committee.vote(2,0,True,forced=True)
    assert not diag['abstain']


def test_nominal_share_and_rejected_mass_not_amplified(tmp_path,monkeypatch):
    c=small();x,y=arrays()
    original_apply=_apply_delta
    captured=[];original_aggregate=__import__('scripts.trust_screen_v11.research.defenses',fromlist=['TwoLayerDefense']).TwoLayerDefense.aggregate
    def aggregate(self,state,updates,context):
        result=original_aggregate(self,state,updates,context)
        total=sum(u.sample_count for u in updates)
        expected={k:sum(u.state[k]*(u.sample_count/total) for u in updates if u.client_id in result.accepted_client_ids)
                  for k,v in updates[0].state.items() if v.is_floating_point()}
        captured.append((result,expected))
        return result
    import scripts.trust_screen_v11.research.runner_v2 as module
    monkeypatch.setattr(module.TwoLayerDefense,'aggregate',aggregate)
    # Deterministically accept one proposal; actual scoring remains separately tested.
    def vote(self,validator,client,corrupt,forced=None):
        from scripts.trust_screen_v11.research.contracts import ValidationVote
        return ValidationVote(validator,client,0.,client==min(self.updates),False,'test'),{'abstain':False}
    monkeypatch.setattr(LossCommittee,'vote',vote)
    def apply(model,delta):
        # Candidate constructions also use _apply_delta; intercept only the
        # immediate final aggregation after aggregate() appended its result.
        if captured:
            _,expected=captured.pop(0)
            for k,v in expected.items():torch.testing.assert_close(delta[k],v)
        return original_apply(model,delta)
    monkeypatch.setattr(module,'_apply_delta',apply)
    summary=BenchmarkRunner(c,tmp_path).run_arrays(x,y,lambda:nn.Linear(3,2))
    assert summary['rounds_completed']==3


def test_probe_scoring_uses_nominal_share_and_counts_work():
    c=small();model=nn.Linear(1,2,bias=False)
    with torch.no_grad():model.weight.zero_()
    delta=torch.tensor([[1.],[-1.]])
    updates=[ClientUpdate(0,{'weight':delta},1),ClientUpdate(1,{'weight':delta*2},3)]
    stub=runner_stub(c);x=np.array([[1.],[2.],[3.]],dtype='float32');y=np.array([0,0,1])
    committee=LossCommittee(stub,model,x,y,SimpleNamespace(validation=[np.arange(3)]*5),updates)
    _,loss=committee.pair(2,0)
    expected=torch.nn.functional.cross_entropy(torch.tensor(x)@(delta*.25).T,torch.tensor(y),reduction='none')
    np.testing.assert_allclose(loss,expected.numpy())
    assert stub.validation_work['private_example_evaluations']==6
    committee.pair(2,0)
    assert stub.validation_work['private_example_evaluations']==6  # cached, no double count


@pytest.mark.parametrize('arm',['rank_original','fedavg','clip_only'])
def test_historical_clean_models_match_v10(tmp_path,arm):
    c=small(arm)
    old=V10Config(**dict(c.to_dict(),protocol_version='trust-mechanism-v10'))
    x,y=arrays()
    a=V10Runner(old,tmp_path/'old').run_arrays(x,y,lambda:nn.Linear(3,2))
    b=BenchmarkRunner(c,tmp_path/'new').run_arrays(x,y,lambda:nn.Linear(3,2))
    traces=[json.loads((tmp_path/n/'chain-trace.json').read_text()) for n in ('old','new')]
    assert [r['model_hash'] for r in traces[0]]==[r['model_hash'] for r in traces[1]]
    assert a['final_utility']==b['final_utility']


@pytest.mark.parametrize('arm',['scaled_quorum','fedval'])
def test_heldout_labels_cannot_change_models(tmp_path,arm):
    c=small(arm);x,y=arrays();ex=x[:20].copy();ey=y[:20].copy()
    for name,labels in [('a',ey),('b',1-ey)]:
        BenchmarkRunner(c,tmp_path/name).run_arrays(x,y,lambda:nn.Linear(3,2),(ex,labels))
    traces=[json.loads((tmp_path/n/'chain-trace.json').read_text()) for n in ('a','b')]
    assert [r['model_hash'] for r in traces[0]]==[r['model_hash'] for r in traces[1]]


def clean_records(tmp_path):
    x,y=arrays();c=replace(small('fedavg'),rounds=1)
    base=BenchmarkRunner(c,tmp_path).run_arrays(x,y,lambda:nn.Linear(3,2))
    rows=[]
    for dataset in study.DATASETS:
        for seed in study.SEEDS:
            for arm in ('fedavg','scaled_quorum'):
                rows.append(dict(study.enrich(base),arm=arm,dataset=dataset,seed=seed,final_utility=.7))
    return rows


def test_gate_requires_six_clean_cells_and_no_relaxation(tmp_path):
    records=clean_records(tmp_path)
    assert study.candidate_gate(records,'scaled_quorum')['passed']
    candidate=next(r for r in records if r['arm']=='scaled_quorum')
    candidate['final_utility']=.649
    assert not study.candidate_gate(records,'scaled_quorum')['passed']
    candidate['final_utility']=.7
    candidate['split_bindings']=dict(candidate['split_bindings'],train='wrong')
    assert not study.candidate_gate(records,'scaled_quorum')['passed']
    assert not study.candidate_gate(records[:-2],'scaled_quorum')['passed']


def test_asr_enrichment_is_post_onset_and_does_not_mutate_summary():
    summary={'configuration':{'attack_start_round':3},'attack':'model_replacement',
        'targeted_attack_success_history':[.01,.02,.7,.9],'clean_trigger_prediction_history':[]}
    enriched=study.enrich(summary)
    assert enriched['post_onset_mean_asr']==pytest.approx(.8)
    assert 'post_onset_mean_asr' not in summary


def test_analysis_emits_only_complete_matched_means_and_seals_detect_tamper(tmp_path,monkeypatch):
    records=clean_records(tmp_path/'cell')
    monkeypatch.setattr(study,'JOB',tmp_path/'report')
    study.analyze(records,{'held':12})
    d=json.loads((study.JOB/'analysis.json').read_text())
    group=next(g for g in d['groups'] if g['arm']=='scaled_quorum' and g['dataset']=='fashion_mnist' and g['attack']=='none')
    assert group['complete'] and group['mean_utility']==pytest.approx(.7)
    records[1]['split_bindings']=dict(records[1]['split_bindings'],train='wrong')
    study.analyze(records,{})
    d=json.loads((study.JOB/'analysis.json').read_text())
    group=next(g for g in d['groups'] if g['arm']=='scaled_quorum' and g['dataset']=='fashion_mnist' and g['attack']=='none')
    assert not group['complete'] and group['mean_utility'] is None
    folder=tmp_path/'cell';study.seal(folder)
    study.load_completed(folder,replace(small('fedavg'),rounds=1))
    (folder/'splits.json').write_text('{}')
    with pytest.raises(ValueError,match='digest mismatch'):study.load_completed(folder,small())


@pytest.mark.parametrize('arm',['fedavg','clip_only','fedval','scaled_quorum'])
def test_targeted_attack_pipeline_and_budgets(tmp_path,arm):
    c=small(arm,'model_replacement');x,y=arrays()
    summary=BenchmarkRunner(c,tmp_path).run_arrays(x,y,lambda:nn.Linear(3,2))
    enriched=study.enrich(summary)
    assert enriched['post_onset_mean_asr'] is not None
    assert summary['malicious_updates_submitted']>0
    if arm=='fedval':assert summary['validation_work']['reference_example_evaluations']>0
    if arm=='scaled_quorum':assert summary['validation_work']['private_example_evaluations']>0
