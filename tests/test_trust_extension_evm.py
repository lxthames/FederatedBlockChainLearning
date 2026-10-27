import json
import os
from dataclasses import replace
import numpy as np
import pytest
import torch
from torch import nn
from scripts.trust_extension.research.config import ExperimentConfig
from scripts.trust_extension.research.runner_v2 import BenchmarkRunner
from scripts.trust_extension.research.evm import EVMAdapter


@pytest.mark.skipif(os.environ.get('TRUST_EVM_TESTS')!='1',reason='requires local Hardhat node')
def test_extension_recovery_trace_agrees_with_contract(tmp_path):
    c=ExperimentConfig(experiment_id='extension-evm-test',dataset='synthetic',num_clients=4,
        participation_rate=1,rounds=3,seed=7,device='cpu',server_samples_per_class=2,
        defense='two_layer',recovery_policy='scheduled_accountable',intervention_round=2,
        intervention_checkpoint=1,minimum_validation_delta=-100,
        minimum_trust=0,trust_floor=0,trust_reward=0,trust_penalty=0)
    rng=np.random.default_rng(5);y=np.tile([0,1],100)
    x=(rng.normal(size=(200,2))+y[:,None]).astype(np.float32)
    result=BenchmarkRunner(c,tmp_path/'memory').run_arrays(x,y,lambda:nn.Linear(2,2))
    trace=json.loads((tmp_path/'memory/chain-trace.json').read_text())
    adapter=EVMAdapter(replace(c,experiment_id=c.experiment_id+'-chain'),tmp_path/'chain')
    adapter.genesis(trace[0])
    for row in trace[1:]:adapter.round(row)
    assert adapter.contract.functions.activeVersion().call()==result['active_version']
    for client,balance in result['final_stakes'].items():
        assert adapter.contract.functions.stake(adapter.account(int(client))).call()==adapter.units(balance)
    assert all(row['status']==1 for row in adapter.receipts)
