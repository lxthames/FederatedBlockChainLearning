import json
import numpy as np
import pytest
import torch
from torch import nn
from scripts.trust_math_lab.loss_voting import paired_decision,rank_approve,clip_updates
from scripts.trust_math_lab.research.contracts import ClientUpdate
from scripts.trust_math_lab.research.config import ExperimentConfig
from scripts.trust_math_lab.research.runner_v2 import BenchmarkRunner
from scripts.trust_math_lab.study import manifest,clean_gate,attack_gate

def test_pairing_removes_shared_loss_variation():
    accept,d=paired_decision([1,10,100],[1.02,10.02,100.02],.01,1.64)
    assert not accept
    assert d['standard_error']==pytest.approx(0,abs=1e-12)
    assert paired_decision([1,10,100],[.9,9.9,99.9],.01,1.64)[0]

def test_insufficient_and_invalid_loss_reports():
    assert not paired_decision([1],[.5],.01,1.64)[0]
    with pytest.raises(ValueError):paired_decision([1,2],[np.nan,3],.01,1.64)

def test_rank_threshold_ties_and_id_permutation():
    scores={4:3.,7:2.,8:2.,9:-4.}
    assert rank_approve(scores,7,.5)[0]
    assert rank_approve(scores,8,.5)[0]
    assert not rank_approve(scores,9,.5)[0]
    assert all(rank_approve({i:1. for i in range(4)},i,.5)[0] for i in range(4))

def test_clipping_is_bounded_and_label_independent():
    updates=[ClientUpdate(i,{'w':torch.tensor([n])},1,is_malicious=(i==0)) for i,n in enumerate((1.,2.,100.))]
    result,d=clip_updates(updates,2)
    assert d['bound']==4.
    assert [u.state['w'].item() for u in result]==pytest.approx([1,2,4])
    assert updates[-1].state['w'].item()==100.

@pytest.mark.parametrize('rule',['paired_loss','scaled_loss','rank_loss'])
def test_loss_rules_independent_of_evaluation_labels(tmp_path,rule):
    c=ExperimentConfig(experiment_id='loss-test',dataset='synthetic',num_clients=4,participation_rate=1,
       rounds=2,seed=11,device='cpu',server_samples_per_class=2,defense='two_layer',vote_rule=rule,
       ablation='validation_only',trust_reward=0,rejection_trust_penalty=0,candidate_clip_ratio=2)
    rng=np.random.default_rng(8);y=np.tile([0,1],100);x=(rng.normal(size=(200,2))+y[:,None]).astype(np.float32)
    traces=[]
    for name,labels in [('a',y),('b',1-y)]:
        s=BenchmarkRunner(c,tmp_path/name).run_arrays(x,y,lambda:nn.Linear(2,2),(x+1,labels))
        assert s['recovery_count']==0
        assert all(t==.5 for t in s['final_trust'].values())
        traces.append(json.loads((tmp_path/name/'chain-trace.json').read_text()))
    assert traces[0]==traces[1]

def test_screening_is_bounded_and_incomplete_groups_cannot_pass():
    rows=manifest()
    assert len(rows)==48
    assert sum(r['stage']=='clean' for r in rows)==24
    assert not clean_gate([],'paired_loss')['passed']
    assert not attack_gate([],'low_and_slow')['passed']
    for row in rows:
        c=ExperimentConfig(**row['config'])
        assert c.evidence_tier=='development'
        assert c.seed not in (13,29,47,71,101)
