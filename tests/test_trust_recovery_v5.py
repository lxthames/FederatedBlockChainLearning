import json
import numpy as np
import pytest
from torch import nn
from scripts.trust_recovery_v5.research.config import ExperimentConfig
from scripts.trust_recovery_v5.research.runner_v2 import BenchmarkRunner
from scripts.trust_recovery_v5.study import manifest,gate


def config(**changes):
    return ExperimentConfig(**(dict(experiment_id='v5-test',dataset='synthetic',num_clients=4,
        participation_rate=1,rounds=8,seed=11,device='cpu',server_samples_per_class=2,
        defense='two_layer',vote_rule='rank_loss',candidate_clip_ratio=2,
        trust_reward=0,rejection_trust_penalty=0,recovery_signal='paired_loss_quorum')|changes))


def arrays():
    rng=np.random.default_rng(5);y=np.tile([0,1],100)
    return (rng.normal(size=(200,2))+y[:,None]).astype(np.float32),y


@pytest.mark.parametrize('approve,expected',[ (True,2),(False,0)])
def test_loss_quorum_independent_of_accuracy_but_requires_benefit(tmp_path,monkeypatch,approve,expected):
    import scripts.trust_recovery_v5.research.runner_v2 as module
    monkeypatch.setattr(BenchmarkRunner,'_score',lambda *args:.5)
    monkeypatch.setattr(module,'checkpoint_benefit',lambda *args:(approve,{'test':'controlled decision'}))
    x,y=arrays();s=BenchmarkRunner(config(),tmp_path).run_arrays(x,y,lambda:nn.Linear(2,2))
    assert s['recovery_count']==expected
    assert s['honest_validator_penalty_amount']==0
    assert all(v==20 for v in s['final_stakes'].values())
    events=[json.loads(line) for line in (tmp_path/'events.jsonl').read_text().splitlines()]
    checks=[e for e in events if e['event_type']=='recovery_confirmation']
    assert [e['round_index'] for e in checks]==[4,8]
    for e in checks:
        p=e['payload'];assert not p['accuracy_detection']['detected']
        assert (len(p['affirmative'])>=p['quorum'])==approve
    trace=json.loads((tmp_path/'chain-trace.json').read_text())
    assert len(trace)==9  # genesis and every round remain after restoration
    if approve:
        assert trace[4]['recovery']['restored_version']==0
        assert trace[8]['recovery']['restored_version']==0
    monitors=[e for e in events if e['event_type']=='monitor']
    for start in (0,4):
        assert all(e['payload']['panel']==monitors[start]['payload']['panel'] for e in monitors[start:start+4])


def test_minority_benefit_cannot_restore(tmp_path,monkeypatch):
    import scripts.trust_recovery_v5.research.runner_v2 as module
    calls=[]
    def minority(*args):
        calls.append(1)
        return len(calls)%3==1,{'test':'one of three approvals'}
    monkeypatch.setattr(module,'checkpoint_benefit',minority)
    x,y=arrays();s=BenchmarkRunner(config(),tmp_path).run_arrays(x,y,lambda:nn.Linear(2,2))
    assert len(calls)==6 and s['recovery_count']==0


def test_pilot_is_bounded_and_preserves_immediate_attack_controls():
    rows=manifest();assert len(rows)==15
    for variant in ('rank_trust','accuracy_guard','loss_quorum'):
        cs=[ExperimentConfig(**r['config']) for r in rows if r['variant']==variant]
        assert len(cs)==5 and all(c.seed==11 and not c.recovery_penalties_enabled for c in cs)
        assert {(c.attack,c.attack_start_round) for c in cs}=={
            ('none',1),('low_and_slow',1),('model_replacement',1),('low_and_slow',21),('model_replacement',21)}
    assert not gate([],'loss_quorum')['passed']
    with pytest.raises(ValueError):config(detector='adjacent')
    with pytest.raises(ValueError):config(recovery_benefit_guard=False)


def test_evaluation_labels_cannot_change_votes_or_recovery(tmp_path):
    x,y=arrays();ex=x[:20]+.031;ey=y[:20]
    for name,labels in [('a',ey),('b',1-ey)]:
        BenchmarkRunner(config(rounds=4),tmp_path/name).run_arrays(x,y,lambda:nn.Linear(2,2),(ex,labels))
    a=json.loads((tmp_path/'a/chain-trace.json').read_text())
    b=json.loads((tmp_path/'b/chain-trace.json').read_text())
    assert a==b
