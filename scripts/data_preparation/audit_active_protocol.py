"""Bounded reproductions for code-review findings; preserves experiment artifacts."""
import json
from pathlib import Path
import tempfile
import numpy as np
from scripts.trust_math_lab.loss_voting import paired_decision
from scripts.trust_math_lab.research.config import ExperimentConfig
from scripts.trust_math_lab.research.evm import EVMAdapter
from scripts.data_preparation.run_rank_reintegration import replay

ROOT=Path(__file__).resolve().parents[2]

def run():
    findings={}
    # Invalid pairing is accepted by NumPy broadcasting instead of rejected.
    result,details=paired_decision([1.,2.,3.],[.5],.01,1.64)
    findings['unequal_loss_lengths_not_rejected']={'reproduced':True,'approved':result,'details':details}
    # A previous agreement file bypasses trace validation on resume.
    with tempfile.TemporaryDirectory() as directory:
        folder=Path(directory);(folder/'evm-replay').mkdir()
        (folder/'chain-trace.json').write_text('[{"changed": true}]')
        (folder/'evm-replay/agreement.json').write_text('{"status":"passed","trace_sha256":"wrong"}')
        replay(None,folder,{})
        findings['cached_replay_agreement_not_validated']={'reproduced':True}
    # Exercise the deployed bytecode, using only new local development accounts/contracts.
    folder=ROOT/'artifacts/code-review-v1/contract-reproduction'
    if folder.exists():raise RuntimeError('Preserve prior audit; choose a new output directory')
    c=ExperimentConfig('code-review-contract', 'synthetic',5,1.,device='cpu')
    adapter=EVMAdapter(c,folder)
    for i in range(5):
        adapter.send(adapter.contract.functions.enroll(adapter.account(i)),adapter.sender,'audit_enroll')
        if i!=0:
            adapter.send(adapter.contract.functions.deposit(),adapter.account(i),'audit_deposit',adapter.units(20))
    adapter.send(adapter.contract.functions.genesis(bytes.fromhex('11'*32)),adapter.sender,'audit_genesis')
    # Proposer 0 has ZERO stake. Four validators imply E=4, specified C=3.
    receipt=adapter.send(adapter.contract.functions.openCandidate(1,adapter.account(0),bytes.fromhex('22'*32),
        [adapter.account(i) for i in range(1,5)]),adapter.sender,'audit_open_understaked_oversized')
    findings['contract_accepts_zero_stake_proposer_and_nonpolicy_size']={
        'reproduced':receipt['status']==1,'proposer_stake':0,'eligible_validators':4,
        'specified_committee_size':3,'accepted_committee_size':4,
        'scope':'Registrar is trusted; membership stake is enforced, proposer stake/size policy are not.'}
    findings['publication_ready']=False
    path=ROOT/'artifacts/code-review-v1/reproductions.json'
    path.write_text(json.dumps(findings,indent=2),encoding='utf-8')
    print(json.dumps(findings,indent=2))

if __name__=='__main__':run()
