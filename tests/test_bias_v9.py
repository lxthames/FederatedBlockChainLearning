from dataclasses import replace
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from threadpoolctl import threadpool_limits
from scripts.trust_bias_v9.loss_voting import LossCommittee
from scripts.trust_bias_v9.research.config import ExperimentConfig
from scripts.trust_bias_v9.research.contracts import ClientUpdate
from scripts.trust_bias_v9.research.runner_v2 import BenchmarkRunner
from scripts.trust_bias_v9 import study
from scripts.trust_literature_v6.research.config import ExperimentConfig as OldConfig
from scripts.trust_literature_v6.research.runner_v2 import BenchmarkRunner as OldRunner


@pytest.fixture(autouse=True)
def bounded_threads():
    with threadpool_limits(limits=2):yield


def small_config(variant='original'):
    return ExperimentConfig(experiment_id=f'test-{variant}',dataset='synthetic',num_clients=5,
        participation_rate=1.,rounds=2,seed=11,defense='two_layer',vote_rule='rank_loss',
        ablation='validation_only',validation_variant=variant,server_samples_per_class=2,
        device='cpu',num_classes=2,candidate_clip_ratio=2.)


def test_original_rule_reproduces_old_committed_models(tmp_path):
    c=small_config();d=c.to_dict();d.pop('validation_variant');d['protocol_version']='trust-literature-v6'
    old=OldConfig(**d)
    rng=np.random.default_rng(4);x=rng.normal(size=(150,3)).astype('float32');y=np.arange(150)%2
    factory=lambda:nn.Linear(3,2)
    a=OldRunner(old,tmp_path/'old').run_arrays(x,y,factory)
    b=BenchmarkRunner(c,tmp_path/'new').run_arrays(x,y,factory)
    def hashes(name):return [r['model_hash'] for r in json.loads((tmp_path/name/'chain-trace.json').read_text())]
    assert hashes('old')==hashes('new') and a['final_utility']==b['final_utility']


def test_probe_uses_nominal_sample_share_and_preserves_submitted_update():
    model=nn.Linear(1,2,bias=False)
    with torch.no_grad():model.weight.zero_()
    delta=torch.tensor([[1.],[-1.]])
    updates=[ClientUpdate(0,{'weight':delta},1),ClientUpdate(1,{'weight':delta*2},3)]
    x=np.array([[1.],[2.],[3.]],dtype='float32');y=np.array([0,0,1])
    c=small_config('probe');runner=SimpleNamespace(config=c,device=torch.device('cpu'))
    committee=LossCommittee(runner,model,x,y,SimpleNamespace(validation=[np.arange(3)]*5),updates)
    b,l=committee.pair(2,0)
    expected=torch.nn.functional.cross_entropy(torch.tensor(x)@(delta*.25).T,torch.tensor(y),reduction='none')
    np.testing.assert_allclose(l,expected.numpy())
    torch.testing.assert_close(committee.models[0].weight,delta*.25)
    torch.testing.assert_close(updates[0].state['weight'],delta)
    macro=LossCommittee(SimpleNamespace(config=replace(c,validation_variant='macro_probe')),model,x,y,
                        SimpleNamespace(validation=[np.arange(3)]*5),updates)
    assert macro.score_mean(2,np.array([1.,3.,8.]))==5.


def test_margin_backoff_approves_uncertain_neutral_but_not_clear_harm():
    model=nn.Linear(1,2,bias=False)
    updates=[ClientUpdate(i,{'weight':torch.ones(2,1)*.01},2) for i in (0,1)]
    c=replace(small_config('margin_probe'),ranking_keep_fraction=.5,loss_tolerance=0.,loss_standard_errors=2.)
    committee=LossCommittee(SimpleNamespace(config=c),model,np.zeros((2,1)),np.array([0,1]),
                            SimpleNamespace(validation=[np.arange(2)]*5),updates)
    pairs={0:(np.ones(2),np.array([1.01,.99])),1:(np.ones(2),np.array([.9,.9]))}
    committee.pair=lambda validator,client:pairs[client]
    vote,diag=committee.vote(2,0,False)
    assert not diag['rank_approved'] and diag['nonharmful_probe'] and vote.approve
    pairs[0]=(np.ones(2),np.array([1.2,1.2]))
    vote,diag=committee.vote(2,0,False)
    assert not diag['rank_approved'] and not diag['nonharmful_probe'] and not vote.approve


@pytest.mark.parametrize('variant',['probe','macro_probe','margin_probe'])
def test_simulator_attacker_flags_do_not_change_votes(variant):
    model=nn.Linear(1,2,bias=False)
    with torch.no_grad():model.weight.zero_()
    c=small_config(variant)
    x=np.array([[1.],[2.],[3.]],dtype='float32');y=np.array([0,0,1])
    updates=[ClientUpdate(i,{'weight':torch.tensor([[.1],[-.1]])*(i+1)},2) for i in (0,1)]
    results=[]
    for flagged in (False,True):
        committee=LossCommittee(SimpleNamespace(config=c,device=torch.device('cpu')),model,x,y,
                                SimpleNamespace(validation=[np.arange(3)]*5),
                                [replace(u,is_malicious=flagged) for u in updates])
        results.append(committee.vote(2,0,False))
    assert results[0]==results[1]


@pytest.mark.parametrize('variant',['probe','macro_probe','margin_probe'])
def test_heldout_labels_do_not_change_votes_or_model_updates(tmp_path,variant):
    c=small_config(variant);rng=np.random.default_rng(7)
    x=rng.normal(size=(150,3)).astype('float32');y=np.arange(150)%2
    ex=rng.normal(size=(20,3)).astype('float32');ey=np.arange(20)%2
    for name,labels in [('a',ey),('b',1-ey)]:
        BenchmarkRunner(c,tmp_path/name).run_arrays(x,y,lambda:nn.Linear(3,2),(ex,labels))
    traces=[json.loads((tmp_path/name/'chain-trace.json').read_text()) for name in ('a','b')]
    assert [r['model_hash'] for r in traces[0]]==[r['model_hash'] for r in traces[1]]
    events=[[json.loads(line) for line in (tmp_path/name/'events.jsonl').read_text().splitlines()] for name in ('a','b')]
    assert [e['payload'] for e in events[0] if e['event_type']=='loss_vote']==[e['payload'] for e in events[1] if e['event_type']=='loss_vote']


def test_group_gate_requires_all_clean_seeds_and_fashion_guard():
    records=[]
    def row(dataset,method,seed):
        return dict(dataset=dataset,method=method,seed=seed,attack='none',final_utility=.7,
                    unsupported_rounds=0,honest_validator_penalty_amount=0)
    for seed in study.SEEDS:
        records.extend([row('femnist','fedavg',seed),row('femnist','probe',seed)])
    assert not study.group_gate(records,'probe')['passed']
    records.extend([row('fashion_mnist','fedavg',11),row('fashion_mnist','probe',11)])
    assert study.group_gate(records,'probe')['passed']
    records[1]['final_utility']=.64
    assert not study.group_gate(records,'probe')['passed']


def test_manifest_has_mandatory_parity_and_clean_first_order():
    rows=study.manifest()
    assert len(rows)==42 and all(r['parity'] for r in rows[:2])
    assert all(not r['parity'] and r['config']['attack']=='none' for r in rows[2:18])
    assert all(r['config']['dataset']=='femnist' and r['config']['attack']!='none' for r in rows[18:])
    control=study.config('clip_only','femnist',11)
    assert control.ranking_keep_fraction==1 and control.trust_reward==control.rejection_trust_penalty==0
    with pytest.raises(ValueError):replace(small_config('probe'),ablation='full')
