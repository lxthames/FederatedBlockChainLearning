import json
import os
from dataclasses import replace
import numpy as np
import pytest
import torch
from torch import nn
from scripts.data_preparation.run_rank_reintegration import (
    manifest,config,validate_reference,fingerprint_config,clean_gate,replay)
from scripts.trust_math_lab.research.config import ExperimentConfig
from scripts.trust_math_lab.research.runner_v2 import BenchmarkRunner


def test_manifest_is_development_and_isolates_mechanisms():
    rows=manifest()
    assert len(rows)==27
    assert sum(r['stage']=='clean' for r in rows)==9
    for row in rows:
        c=ExperimentConfig(**row['config'])
        assert c.seed in (11,17,23) and c.evidence_tier=='development'
        assert c.vote_rule=='rank_loss' and c.candidate_clip_ratio==2
        if row['variant']=='recovery_only':
            assert c.ablation=='no_penalties' and c.trust_reward==0 and c.rejection_trust_penalty==0
        elif row['variant']=='trust_only':
            assert c.ablation=='validation_only' and c.trust_reward>0
        else:
            assert c.ablation=='full' and c.trust_reward>0


def test_reference_reuse_requires_exact_configuration():
    c=config('trust_only',11,'none').to_dict()
    s={'status':'completed','configuration':c,'config_hash':fingerprint_config(c)}
    validate_reference(s,c)
    with pytest.raises(ValueError):validate_reference(s,c|{'rounds':100})
    with pytest.raises(ValueError):validate_reference(s|{'config_hash':'bad'},c)


def gate_records(unsupported=0,penalties=0):
    refs=[];records=[]
    for seed in (11,17,23):
        r=dict(seed=seed,attack='none',variant='rank_only',final_utility=.6)
        refs.append(r)
        records.append(dict(r,variant='combined',rounds_completed=40,num_clients=50,
            configuration={'initial_stake':20},unsupported_rounds=unsupported,
            honest_validator_penalty_amount=penalties))
    return records,refs


def test_clean_gate_rejects_committee_collapse_or_excess_honest_penalties():
    records,refs=gate_records()
    assert clean_gate(records,refs,'combined')['passed']
    records,refs=gate_records(unsupported=10)
    assert not clean_gate(records,refs,'combined')['passed']
    records,refs=gate_records(penalties=90)
    assert not clean_gate(records,refs,'combined')['passed']
    assert not clean_gate(records[:-1],refs,'combined')['passed']


def synthetic_config(ablation):
    return ExperimentConfig(experiment_id='rank-recovery-test',dataset='synthetic',num_clients=4,
        participation_rate=1,rounds=3,seed=11,device='cpu',server_samples_per_class=2,
        defense='two_layer',vote_rule='rank_loss',candidate_clip_ratio=2,ablation=ablation,
        recovery_policy='scheduled_accountable',intervention_round=2,intervention_checkpoint=1)


@pytest.mark.parametrize('ablation',['no_penalties','full'])
def test_rank_restoration_preserves_expected_economics(tmp_path,ablation):
    c=synthetic_config(ablation)
    rng=np.random.default_rng(5);y=np.tile([0,1],100)
    x=(rng.normal(size=(200,2))+y[:,None]).astype(np.float32)
    s=BenchmarkRunner(c,tmp_path).run_arrays(x,y,lambda:nn.Linear(2,2))
    trace=json.loads((tmp_path/'chain-trace.json').read_text())
    assert s['recovery_count']==1
    assert trace[2]['recovery']['restored_version']==1
    assert trace[3]['parent_version']==1
    if ablation=='no_penalties':
        assert all(v==20 for v in s['final_stakes'].values())
        assert s['honest_validator_penalty_amount']==0
    else:
        assert any(v<20 for v in s['final_stakes'].values())
        assert s['honest_validator_penalty_amount']>0


@pytest.mark.skipif(os.environ.get('TRUST_EVM_TESTS')!='1',reason='requires local Hardhat')
def test_rank_reintegration_trace_contract_agreement(tmp_path):
    c=synthetic_config('full')
    rng=np.random.default_rng(5);y=np.tile([0,1],100)
    x=(rng.normal(size=(200,2))+y[:,None]).astype(np.float32)
    s=BenchmarkRunner(c,tmp_path).run_arrays(x,y,lambda:nn.Linear(2,2))
    replay(c,tmp_path,s)
    agreement=json.loads((tmp_path/'evm-replay/agreement.json').read_text())
    assert agreement['status']=='passed' and agreement['stake_agreement']
    assert agreement['transaction_count']>0 and agreement['gas_used']>0
