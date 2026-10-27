import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
from scripts.trust_recovery_v4.loss_voting import paired_decision
from scripts.trust_recovery_v4.recovery_guard import checkpoint_benefit
from scripts.trust_recovery_v4.runtime import exclusive_lock,replay,verify_cache
from scripts.trust_recovery_v4.research.config import ExperimentConfig
from scripts.trust_recovery_v4.research.runner_v2 import BenchmarkRunner
from scripts.trust_recovery_v4.study import manifest,gate

@pytest.mark.parametrize('a,b',[([1,2,3],[.5]),(1,2),([[1,2]],[[1,2]]),([],[]),([1,float('inf')],[1,2])])
def test_invalid_paired_arrays_rejected(a,b):
    with pytest.raises(ValueError):paired_decision(a,b,.01,1.64)

def test_benefit_requires_checkpoint_to_be_better():
    assert checkpoint_benefit([2,3,4],[1,2,3],.02,2)[0]
    assert not checkpoint_benefit([1,2,3],[2,3,4],.02,2)[0]
    assert not checkpoint_benefit([1],[0],.02,2)[0]
    assert not checkpoint_benefit([1,2,3],[1,2,3],.02,2)[0]
    # High mean benefit but large sampling uncertainty should not suffice.
    assert not checkpoint_benefit([10,1],[1,2],.02,2)[0]

def test_exclusive_coordinator_lock_releases_after_error(tmp_path):
    lock=tmp_path/'lock'
    with pytest.raises(ValueError):
        with exclusive_lock(lock):
            with pytest.raises(RuntimeError):
                with exclusive_lock(lock):pass
            raise ValueError('simulate interruption')
    with exclusive_lock(lock):pass

def test_cached_agreement_checks_binding_and_receipts(tmp_path):
    p=tmp_path/'transactions.jsonl';p.write_text('{"status":1}\n')
    binding={'trace_sha256':'current'}
    agreement={'status':'passed','binding':binding,'transactions_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    verify_cache(agreement,binding,tmp_path)
    with pytest.raises(ValueError):verify_cache(agreement,{'trace_sha256':'changed'},tmp_path)
    p.write_text('{"status":0}\n')
    with pytest.raises(ValueError):verify_cache(agreement,binding,tmp_path)

def test_manifest_and_clean_penalty_gate():
    rows=manifest();assert len(rows)==27
    assert sum(r['config']['attack']=='none' for r in rows)==9
    assert not gate([],'guarded_penalties')['passed']
    for r in rows:
        c=ExperimentConfig(**r['config'])
        assert c.recovery_benefit_guard and c.seed in (11,17,23)
        assert c.recovery_penalties_enabled==(r['variant']=='guarded_penalties')

def cfg(**changes):
    return ExperimentConfig(**(dict(experiment_id='v4-regression',dataset='synthetic',num_clients=4,
        participation_rate=1,rounds=3,seed=11,device='cpu',server_samples_per_class=2,
        defense='two_layer',vote_rule='rank_loss',candidate_clip_ratio=2,
        recovery_policy='scheduled_accountable',intervention_round=2,intervention_checkpoint=1)|changes))

def arrays():
    rng=np.random.default_rng(5);y=np.tile([0,1],100)
    return (rng.normal(size=(200,2))+y[:,None]).astype(np.float32),y

def test_penalties_disabled_by_default_and_all_code_fingerprinted(tmp_path):
    c=cfg();x,y=arrays()
    s=BenchmarkRunner(c,tmp_path).run_arrays(x,y,lambda:nn.Linear(2,2))
    assert s['recovery_count']==1
    assert s['honest_validator_penalty_amount']==0
    assert all(v==20 for v in s['final_stakes'].values())
    assert 'loss_voting.py' in s['source_fingerprints']
    assert 'recovery_guard.py' in s['source_fingerprints']

def test_alarm_cannot_restore_without_checkpoint_benefit_quorum(tmp_path,monkeypatch):
    import scripts.trust_recovery_v4.research.runner_v2 as module
    from scripts.trust_recovery_v4.research.detector import DetectionResult
    class AlwaysAlarm:
        def __init__(self,*args):pass
        def reset(self):pass
        def observe(self,value):return DetectionResult(True,'test_alarm',.2,-.2)
    monkeypatch.setattr(module,'TrendDetector',AlwaysAlarm)
    monkeypatch.setattr(module,'checkpoint_benefit',lambda *args:(False,{'test':'checkpoint is worse'}))
    c=cfg(recovery_policy='protocol');x,y=arrays()
    s=BenchmarkRunner(c,tmp_path).run_arrays(x,y,lambda:nn.Linear(2,2))
    assert s['recovery_count']==0
    assert s['honest_validator_penalty_amount']==0
    assert 'recovery_confirmation' in (tmp_path/'events.jsonl').read_text()

@pytest.mark.skipif(os.environ.get('TRUST_EVM_TESTS')!='1',reason='requires Hardhat')
def test_contract_enforces_stake_snapshot_size_and_membership(tmp_path):
    from scripts.trust_recovery_v4.research.evm import EVMAdapter
    c=cfg(num_clients=5);a=EVMAdapter(c,tmp_path)
    for i in range(5):
        a.send(a.contract.functions.enroll(a.account(i)),a.sender,'enroll')
        if i:a.send(a.contract.functions.deposit(),a.account(i),'deposit',a.units(20))
    a.send(a.contract.functions.genesis(bytes.fromhex('11'*32)),a.sender,'genesis')
    eligible=[a.account(i) for i in range(1,5)]
    def candidate(members,pool=eligible):
        return a.contract.functions.openCandidate(1,a.account(0),bytes.fromhex('22'*32),members,pool)
    with pytest.raises(Exception):candidate(eligible[:3]).transact({'from':a.sender})
    a.send(a.contract.functions.deposit(),a.account(0),'deposit',a.units(20))
    with pytest.raises(Exception):candidate(eligible).transact({'from':a.sender})
    with pytest.raises(Exception):candidate(eligible[:3],[eligible[0]]*4).transact({'from':a.sender})
    with pytest.raises(Exception):candidate(eligible[:3],[a.account(0)]+eligible).transact({'from':a.sender})
    a.send(candidate(eligible[:3]),a.sender,'valid_candidate')

@pytest.mark.skipif(os.environ.get('TRUST_EVM_TESTS')!='1',reason='requires Hardhat')
def test_replay_revalidates_live_state_and_rejects_changed_trace(tmp_path):
    c=cfg(recovery_penalties_enabled=True);x,y=arrays()
    s=BenchmarkRunner(c,tmp_path).run_arrays(x,y,lambda:nn.Linear(2,2))
    a=replay(c,tmp_path,s)
    assert a['status']=='passed'
    assert replay(c,tmp_path,s)==a
    trace=tmp_path/'chain-trace.json'
    trace.write_text(trace.read_text()+'\n')
    with pytest.raises(ValueError,match='Stale'):replay(c,tmp_path,s)
