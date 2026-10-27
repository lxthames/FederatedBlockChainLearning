from dataclasses import replace
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from threadpoolctl import threadpool_limits

from scripts.trust_mechanism_v10.research.config import ExperimentConfig
from scripts.trust_mechanism_v10.research.contracts import ClientUpdate, ValidationVote, VoteDecision
from scripts.trust_mechanism_v10.research.defenses import TwoLayerDefense, AggregationContext
from scripts.trust_mechanism_v10.research.validation import VoteAggregator
from scripts.trust_mechanism_v10.research.runner_v2 import BenchmarkRunner
from scripts.trust_mechanism_v10.research.splits import DataSplit, matched_test_shards, freeze_evaluation_and_redistribute
from scripts.trust_mechanism_v10.loss_voting import LossCommittee
from scripts.trust_mechanism_v10.pairing import compare_pair, split_bindings
from scripts.trust_mechanism_v10 import study
from scripts.trust_bias_v9.research.config import ExperimentConfig as V9Config
from scripts.trust_bias_v9.research.runner_v2 import BenchmarkRunner as V9Runner


@pytest.fixture(autouse=True)
def threads():
    with threadpool_limits(limits=2):
        torch.set_num_threads(2)
        yield


def config(**kwargs):
    values = dict(experiment_id='test-v10',dataset='synthetic',num_clients=4,
        participation_rate=1.,rounds=4,seed=11,defense='two_layer',vote_rule='rank_loss',
        ablation='validation_only',server_samples_per_class=2,device='cpu',num_classes=2,
        candidate_clip_ratio=2.,ranking_keep_fraction=1.)
    return ExperimentConfig(**dict(values,**kwargs))


def data():
    x = np.random.default_rng(71).normal(size=(160,3)).astype('float32')
    return x, (x[:,0]>0).astype('int64')


def vote(i, approve):
    return ValidationVote(i,9,0.,approve,False,'test')


@pytest.mark.parametrize('votes,outcome', [([], 'insufficient_response'),
    ([vote(0,True)], 'insufficient_response'),
    ([vote(0,False)], 'insufficient_response'),
    ([vote(0,False),vote(1,False)], 'quality_rejected'),
    ([vote(0,True),vote(1,True)], 'accepted'),
    ([vote(0,True),vote(1,False),vote(2,False)], 'quality_rejected')])
def test_response_outcomes(votes,outcome):
    decision=VoteAggregator().decide(votes,2,[0,1,2])
    assert decision.outcome==outcome and decision.response_count==len(votes)
    assert decision.expected_count==3


def test_committee_vote_integrity():
    for votes,members in [([vote(0,True),vote(0,False)],[0,1,2]),
                          ([vote(4,True)],[0,1,2]),([], [0,0,1])]:
        with pytest.raises(ValueError): VoteAggregator().decide(votes,2,members)


def test_no_response_all_candidate_noop_preserves_trust():
    updates=[ClientUpdate(9,{'w':torch.tensor([1.])},1)]
    decision=VoteAggregator().decide([],2,[0,1,2])
    defense=TwoLayerDefense(trust_penalty=.02)
    result=defense.aggregate({},updates,AggregationContext(1,{9:decision}))
    assert defense.trust_score(9)==.5 and not result.accepted_client_ids
    assert result.state['w'].item()==0


def test_withholding_no_longer_destroys_honest_eligibility(tmp_path):
    x,y=data()
    base=config(malicious_validator_fraction=.5,validator_behavior='withhold',
                rejection_trust_policy='rank_rejection')
    summaries={}
    for label,legacy,eligibility in [('fixed',False,'dynamic'),('historical',True,'dynamic'),
                                      ('weights_only',True,'fixed')]:
        c=replace(base,legacy_inconclusive_penalty=legacy,trust_eligibility=eligibility)
        summaries[label]=BenchmarkRunner(c,tmp_path/label).run_arrays(x,y,lambda:nn.Linear(3,2))
    fixed,old,weights = (summaries[n] for n in ('fixed','historical','weights_only'))
    assert fixed['malicious_client_ids']==[]
    for client in (2,3):
        assert fixed['final_trust'][str(client)]==.5
        assert fixed['inconclusive_counts'][client]==4
        assert old['final_trust'][str(client)]<.45
        assert weights['final_trust'][str(client)]<.45
        assert weights['client_eligible_round_counts'][client]==4
    assert fixed['unsupported_rounds']==0 and old['unsupported_rounds']==1
    assert weights['unsupported_rounds']==0


def test_original_clean_training_parity(tmp_path):
    c=config(num_clients=6,rounds=3,ranking_keep_fraction=.8,rejection_trust_policy='rank_rejection')
    fields=V9Config.__dataclass_fields__
    old=V9Config(**{**{k:v for k,v in c.to_dict().items() if k in fields},'protocol_version':'trust-bias-v9'})
    x,y=data()
    a=V9Runner(old,tmp_path/'v9').run_arrays(x,y,lambda:nn.Linear(3,2))
    b=BenchmarkRunner(c,tmp_path/'v10').run_arrays(x,y,lambda:nn.Linear(3,2))
    traces=[json.loads((tmp_path/name/'chain-trace.json').read_text()) for name in ('v9','v10')]
    assert [r['model_hash'] for r in traces[0]]==[r['model_hash'] for r in traces[1]]
    assert a['final_trust']==b['final_trust'] and a['final_utility']==b['final_utility']


def test_validation_redistribution_cannot_move_evaluation_groups():
    y=np.array([0]*10+[1]*10+[0]*40+[1]*40)
    ty=np.array([0]*100+[1]*100)
    original=DataSplit([np.arange(10),np.arange(10,20)],
        [np.arange(20,60),np.arange(60,100)],np.array([],dtype=int),np.array([],dtype=int))
    a=freeze_evaluation_and_redistribute(copy.deepcopy(original),y,ty,config())
    b=freeze_evaluation_and_redistribute(copy.deepcopy(original),y,ty,config(validation_distribution='redistributed'))
    for left,right in zip(matched_test_shards(y,ty,a,11),matched_test_shards(y,ty,b,11)):
        np.testing.assert_array_equal(left,right)
    assert split_bindings(a)['validation']!=split_bindings(b)['validation']
    for key in ('train','evaluation_groups','validation_pool','validation_counts'):
        assert split_bindings(a)[key]==split_bindings(b)[key]
    # Natural writer IDs retain the same assignment too.
    original.natural_test_clients=[np.arange(100),np.arange(100,200)]
    natural=freeze_evaluation_and_redistribute(original,y,ty,config(validation_distribution='redistributed'))
    for a,b in zip(natural.natural_test_clients,natural.evaluation_groups):
        np.testing.assert_array_equal(a,b)


def test_pairing_and_redistribution_end_to_end(tmp_path,monkeypatch):
    x,y=data()
    a=BenchmarkRunner(config(defense='fedavg',rounds=1),tmp_path/'a').run_arrays(x,y,lambda:nn.Linear(3,2))
    b=BenchmarkRunner(config(defense='fedavg',rounds=1,validation_distribution='redistributed'),tmp_path/'b').run_arrays(x,y,lambda:nn.Linear(3,2))
    assert a['final_utility']==b['final_utility']
    assert a['client_test_utilities']==b['client_test_utilities']
    assert not compare_pair(a,b)['matched']
    assert compare_pair(a,b,allow_validation_redistribution=True)['matched']
    monkeypatch.setattr(study,'JOB',tmp_path/'analysis')
    records=[]
    for arm,summary,utility in [('rank_private_dynamic',a,.6),('rank_private_fixed',a,.7),
                               ('rank_redistributed_dynamic',b,.8),('rank_redistributed_fixed',b,.75)]:
        records.append(dict(summary,arm=arm,dataset='fashion_mnist',final_utility=utility))
    study.analyze(records,{})
    analysis=json.loads((study.JOB/'analysis.json').read_text())
    interaction=next(r for r in analysis['contrasts'] if r['contrast']=='validation_by_trust_interaction')
    assert interaction['utility_difference']==pytest.approx(-.15)
    assert all(g['mean_utility_difference'] is None for g in analysis['contrast_groups'])
    records[-1]['split_bindings']=dict(records[-1]['split_bindings'],train='wrong')
    study.analyze(records,{})
    analysis=json.loads((study.JOB/'analysis.json').read_text())
    assert not any(r['contrast']=='validation_by_trust_interaction' for r in analysis['contrasts'])
    assert not compare_pair({}, {},allow_validation_redistribution=True)['matched']
    for key in ('train','evaluation_groups','validation_pool'):
        bad=copy.deepcopy(b);bad['split_bindings'][key]='changed'
        assert not compare_pair(a,bad,allow_validation_redistribution=True)['matched']
    for key in ('initial_model_hash','training_binding','source_hash'):
        bad=copy.deepcopy(b);bad[key]='changed'
        assert not compare_pair(a,bad,allow_validation_redistribution=True)['matched']


def test_rank_signals_are_separate_and_no_penalty_is_default(tmp_path):
    c=config(num_clients=6,ranking_keep_fraction=.8,ranking_norm_penalty=0.)
    updates=[ClientUpdate(i,{'weight':torch.ones(2,1)*.01},1) for i in range(5)]
    committee=LossCommittee(SimpleNamespace(config=c),nn.Linear(1,2,bias=False),
        np.ones((2,1)),np.array([0,1]),SimpleNamespace(validation=[np.arange(2)]*6),updates)
    committee.pair=lambda validator,client:(np.ones(2),np.ones(2)-(.5-.1*client))
    v,d=committee.vote(5,4,False)
    assert not v.approve and d['relative_rank_rejected'] and not d['locally_harmful']
    assert not d['establishes_misconduct']
    committee.pair=lambda validator,client:(np.ones(2),np.ones(2)+(.1+.1*client))
    v,d=committee.vote(5,0,False)
    assert v.approve and d['locally_harmful']
    x,y=data()
    s=BenchmarkRunner(replace(c,rounds=3),tmp_path/'safe').run_arrays(x,y,lambda:nn.Linear(3,2))
    assert c.rejection_trust_policy=='none'
    assert min(s['final_trust'].values())>=c.initial_trust


def test_trust_weighting_can_be_fixed_while_scores_differ():
    updates=[ClientUpdate(0,{'w':torch.tensor([1.])},1),ClientUpdate(1,{'w':torch.tensor([-1.])},1)]
    decisions={i:VoteDecision(True,2,(2,3),0.,outcome='accepted') for i in (0,1)}
    values={}
    for mode in ('fixed','dynamic'):
        defense=TwoLayerDefense(trust_reward=0,trust_penalty=0,trust_weighting=mode)
        defense.set_trust_score(0,.8);defense.set_trust_score(1,.2)
        values[mode]=defense.aggregate({},updates,AggregationContext(1,decisions)).state['w'].item()
    assert values['fixed']==0 and values['dynamic']==pytest.approx(.6)


def test_completed_artifacts_fail_closed_on_mutation(tmp_path):
    x,y=data();c=config(rounds=1)
    BenchmarkRunner(c,tmp_path).run_arrays(x,y,lambda:nn.Linear(3,2))
    study.seal(tmp_path)
    assert study.load_completed(tmp_path,c)['status']=='completed'
    (tmp_path/'splits.json').write_text('{}')
    with pytest.raises(ValueError,match='hash mismatch'):study.load_completed(tmp_path,c)


def test_heldout_labels_cannot_change_training_after_split_freeze(tmp_path):
    x,y=data();ex=x[:20].copy();ey=y[:20].copy()
    c=config(rounds=2,validation_distribution='redistributed',num_clients=6,ranking_keep_fraction=.8)
    for name,labels in [('a',ey),('b',1-ey)]:
        BenchmarkRunner(c,tmp_path/name).run_arrays(x,y,lambda:nn.Linear(3,2),(ex,labels))
    traces=[json.loads((tmp_path/name/'chain-trace.json').read_text()) for name in ('a','b')]
    assert [r['model_hash'] for r in traces[0]]==[r['model_hash'] for r in traces[1]]
    events=[[json.loads(line) for line in (tmp_path/name/'events.jsonl').read_text().splitlines()] for name in ('a','b')]
    assert ([e['payload'] for e in events[0] if e['event_type']=='loss_vote']==
            [e['payload'] for e in events[1] if e['event_type']=='loss_vote'])


def test_finite_manifest_has_channel_control_and_no_automatic_attacks():
    rows=study.manifest()
    assert len(rows)==42 and len({r['config']['experiment_id'] for r in rows})==42
    assert all(r['config']['attack']=='none' for r in rows)
    bridge=study.config('femnist','rank_private_weights_only',11)
    assert bridge.trust_weighting=='dynamic' and bridge.trust_eligibility=='fixed'
    assert study.config('femnist','rank_private_dynamic',11).rejection_trust_policy=='rank_rejection'
    with pytest.raises(ValueError):config(ledger_backend='evm')
    with pytest.raises(ValueError):config(ablation='full')
