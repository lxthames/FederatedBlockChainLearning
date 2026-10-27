"""Run against the local deployment with TRUST_EVM_TESTS=1."""
import os
from pathlib import Path
import pytest
from scripts.research.config import ExperimentConfig

pytestmark=pytest.mark.skipif(os.getenv('TRUST_EVM_TESTS')!='1',reason='requires local Hardhat node')


def test_real_contract_votes_replays_and_recovery(tmp_path):
    from scripts.research.evm import EVMAdapter
    from web3.exceptions import ContractLogicError
    c=ExperimentConfig(experiment_id='contract-invariants',dataset='synthetic',num_clients=5,
        participation_rate=1.0,ledger_backend='evm')
    adapter=EVMAdapter(c,tmp_path)
    adapter.genesis({'enrolled':list(range(5)),'model_hash':'aa'*32})
    chain=adapter.contract
    sender=adapter.sender
    accounts=[adapter.account(i) for i in range(5)]
    def rejects(fn,sender):
        with pytest.raises(Exception): fn.transact({'from':sender})
    rejects(chain.functions.enroll(accounts[0]),accounts[1])
    rejects(chain.functions.openCandidate(1,accounts[0],b'b'*32,[accounts[0],accounts[1],accounts[2]]),sender)
    rejects(chain.functions.openCandidate(1,accounts[0],b'b'*32,[accounts[1],accounts[1],accounts[2]]),sender)
    rejects(chain.functions.openCandidate(1,accounts[0],b'b'*32,[accounts[1],accounts[2]]),sender)
    key=chain.functions.candidateKey(1,0,accounts[0],b'b'*32).call()
    adapter.send(chain.functions.openCandidate(1,accounts[0],b'b'*32,accounts[1:4]),sender,'open_candidate')
    rejects(chain.functions.vote(key,True),accounts[4])
    adapter.send(chain.functions.vote(key,True),accounts[1],'vote')
    rejects(chain.functions.vote(key,True),accounts[1])
    rejects(chain.functions.commitModel(1,b'c'*32,[key]),sender)
    adapter.send(chain.functions.vote(key,True),accounts[2],'vote')
    adapter.send(chain.functions.vote(key,False),accounts[3],'vote')
    assert chain.functions.accepted(key).call()
    adapter.send(chain.functions.commitModel(1,b'c'*32,[key]),sender,'commit_model')
    rejects(chain.functions.vote(key,True),accounts[4])
    rejects(chain.functions.commitModel(1,b'c'*32,[key]),sender)
    recovery=b'r'*32
    adapter.send(chain.functions.openRecovery(recovery,1,1,0,b'd'*32,accounts[1:4],True),sender,'open_recovery')
    rejects(chain.functions.applyRecovery(recovery),sender)
    for account in accounts[1:3]: adapter.send(chain.functions.attestRecovery(recovery),account,'attest')
    rejects(chain.functions.attestRecovery(recovery),accounts[1])
    block_before=adapter.web3.eth.block_number
    adapter.send(chain.functions.applyRecovery(recovery),sender,'recover')
    assert adapter.web3.eth.block_number>block_before
    assert chain.functions.activeVersion().call()==0
    assert chain.functions.modelHash(1).call()==b'c'*32
    assert chain.functions.stake(accounts[1]).call()==adapter.units(15)
    assert chain.functions.stake(accounts[2]).call()==adapter.units(15)
    assert chain.functions.stake(accounts[3]).call()==adapter.units(20)
    rejects(chain.functions.applyRecovery(recovery),sender)
    assert chain.functions.stake(accounts[1]).call()==adapter.units(15)
    assert all(receipt['status']==1 for receipt in adapter.receipts)
