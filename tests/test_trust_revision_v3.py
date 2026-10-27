import json
import os
from dataclasses import replace
import numpy as np
import pytest
import torch
from torch import nn
from scripts.trust_revision_v3.research.config import ExperimentConfig
from scripts.trust_revision_v3.research.validation import LocalValidator
from scripts.trust_revision_v3.research.defenses import TwoLayerDefense, AggregationContext
from scripts.trust_revision_v3.research.contracts import ClientUpdate, VoteDecision
from scripts.trust_revision_v3.research.runner_v2 import BenchmarkRunner
from scripts.trust_revision_v3.development import split_training_pool

def test_absolute_votes_and_adversary_label_independence():
    validator=LocalValidator(1,absolute_tolerance=0.02)
    assert validator.vote(2,.01,0,False).approve
    assert validator.vote(2,.50,.48,False).approve
    assert not validator.vote(2,.50,.47,False).approve
    assert validator.vote(2,.50,.47,True).approve==validator.vote(2,.50,.47,False).approve

def test_all_rejected_updates_trust_and_returns_zero_delta():
    defense=TwoLayerDefense(initial_trust=.5,trust_penalty=.02)
    update=ClientUpdate(0,{'w':torch.tensor([2.])},3)
    decision=VoteDecision(False,2,(),-.3)
    result=defense.aggregate({'w':torch.tensor([7.])},[update],AggregationContext(1,{0:decision}))
    assert result.state['w'].item()==0
    assert not result.accepted_client_ids
    assert defense.trust_score(0)==pytest.approx(.48)

def test_zero_trust_approved_probation_can_recover():
    defense=TwoLayerDefense(initial_trust=0,trust_reward=.02,probation_weight_floor=.1)
    updates=[ClientUpdate(i,{'w':torch.tensor([float(i+1)])},1) for i in (0,1)]
    decisions={i:VoteDecision(True,2,(2,3),0) for i in (0,1)}
    result=defense.aggregate({},updates,AggregationContext(1,decisions))
    assert result.state['w'].item()==pytest.approx(1.5)
    assert defense.trust_score(0)==pytest.approx(.02)

def test_training_only_development_split_disjoint():
    x=np.arange(40)[:,None].astype(np.float32);y=np.arange(40)%2
    split={'train':[list(range(0,15)),list(range(15,30))],
        'validation':[[30,31],[32,33]],'reference':list(range(34,40)),
        'writers':['a','b'],'reference_writers':['c']}
    a,b,ex,ey,s=split_training_pool(x,y,split,11)
    assert set(a[:,0]).isdisjoint(ex[:,0])
    assert len(a)+len(ex)==len(x)
    assert sorted(sum(s['train']+s['validation']+[s['reference']],[]))==list(range(len(a)))
    assert sum(map(len,s['test_clients']))==len(ex)

def config(**kwargs):
    return ExperimentConfig(**(dict(experiment_id='v3-test',dataset='synthetic',num_clients=4,
        participation_rate=1,rounds=3,seed=11,device='cpu',server_samples_per_class=2,
        defense='two_layer',minimum_trust=0,trust_floor=0,validation_absolute_tolerance=.99)|kwargs))

def test_development_evaluation_labels_do_not_change_decisions(tmp_path):
    rng=np.random.default_rng(4);y=np.tile([0,1],100)
    x=(rng.normal(size=(200,2))+y[:,None]).astype(np.float32)
    traces=[]
    for name,ey in [('a',y),('b',1-y)]:
        BenchmarkRunner(config(),tmp_path/name).run_arrays(x,y,lambda:nn.Linear(2,2),(x+1,ey))
        traces.append(json.loads((tmp_path/name/'chain-trace.json').read_text()))
    assert traces[0]==traces[1]

def test_stake_depletion_stops_future_proposals(tmp_path):
    c=config(recovery_policy='scheduled_accountable',intervention_round=2,
             intervention_checkpoint=1,slashing_penalty=15)
    rng=np.random.default_rng(5);y=np.tile([0,1],100)
    x=(rng.normal(size=(200,2))+y[:,None]).astype(np.float32)
    s=BenchmarkRunner(c,tmp_path).run_arrays(x,y,lambda:nn.Linear(2,2))
    trace=json.loads((tmp_path/'chain-trace.json').read_text())
    assert trace[2]['recovery'] is not None
    depleted={int(i) for i,b in s['final_stakes'].items() if b<c.minimum_stake}
    assert depleted
    assert not depleted & {r['proposer'] for r in trace[3]['candidates']}

@pytest.mark.skipif(os.environ.get('TRUST_EVM_TESTS')!='1',reason='requires local Hardhat')
def test_v3_recovery_contract_agreement(tmp_path):
    from scripts.trust_revision_v3.research.evm import EVMAdapter
    c=config(recovery_policy='scheduled_accountable',intervention_round=2,intervention_checkpoint=1)
    rng=np.random.default_rng(5);y=np.tile([0,1],100)
    x=(rng.normal(size=(200,2))+y[:,None]).astype(np.float32)
    s=BenchmarkRunner(c,tmp_path/'memory').run_arrays(x,y,lambda:nn.Linear(2,2))
    trace=json.loads((tmp_path/'memory/chain-trace.json').read_text())
    adapter=EVMAdapter(replace(c,experiment_id='v3-contract-test'),tmp_path/'chain')
    adapter.genesis(trace[0])
    for record in trace[1:]:adapter.round(record)
    assert adapter.contract.functions.activeVersion().call()==s['active_version']
    for client,stake in s['final_stakes'].items():
        assert adapter.contract.functions.stake(adapter.account(int(client))).call()==adapter.units(stake)
    assert all(r['status']==1 for r in adapter.receipts)
