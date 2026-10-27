"""Exclusive coordinator lock and verified local-EVM trace replay cache."""
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import time
from .femnist import dump,digest

@contextmanager
def exclusive_lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    handle=path.open('a+b')
    handle.seek(0,2)
    if handle.tell()==0:handle.write(b'0');handle.flush()
    handle.seek(0)
    try:
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:
        handle.close();raise RuntimeError('Coordinator already holds the experiment lock') from None
    try:yield
    finally:
        handle.seek(0)
        if os.name=='nt':msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
        else:fcntl.flock(handle.fileno(),fcntl.LOCK_UN)
        handle.close()

def replay_binding(c,output,summary):
    output=Path(output)
    config_hash=hashlib.sha256(json.dumps(c.to_dict(),sort_keys=True).encode()).hexdigest()
    if summary.get('config_hash')!=config_hash:raise ValueError('Replay configuration mismatch')
    trace=json.loads((output/'chain-trace.json').read_text())
    versions={r.get('round',0):r['model_hash'] for r in trace}
    if summary['active_version'] not in versions:raise ValueError('Missing active model commitment')
    root=Path(__file__).resolve().parents[2]
    return {'trace_sha256':digest(output/'chain-trace.json'),'config_hash':config_hash,
        'summary_sha256':digest(output/'summary.json'),
        'contract_artifact_sha256':digest(root/'blockchain/build/TrustLedgerV4.json'),
        'experiment_id':c.experiment_id+'-replay','active_version':summary['active_version'],
        'active_model_hash':versions[summary['active_version']],
        'final_stakes':summary['final_stakes']}

def verify_cache(agreement,binding,folder):
    if agreement.get('status')!='passed' or agreement.get('binding')!=binding:
        raise ValueError('Stale or invalid replay agreement')
    if agreement.get('transactions_sha256')!=digest(Path(folder)/'transactions.jsonl'):
        raise ValueError('Replay transaction log changed')
    rows=[json.loads(line) for line in (Path(folder)/'transactions.jsonl').read_text().splitlines()]
    if not rows or any(r['status']!=1 for r in rows):raise ValueError('Invalid transaction receipts')

def verify_live_contract(web3,contract,c,binding):
    if web3.eth.chain_id!=31337:raise ValueError('Wrong replay chain')
    expected=hashlib.sha256(binding['experiment_id'].encode()).digest()
    if bytes(contract.functions.experiment().call())!=expected:raise ValueError('Replay experiment mismatch')
    if contract.functions.activeVersion().call()!=binding['active_version']:raise ValueError('Active version mismatch')
    if bytes(contract.functions.modelHash(binding['active_version']).call()).hex()!=binding['active_model_hash']:
        raise ValueError('Active model hash mismatch')
    for client,stake in binding['final_stakes'].items():
        account=web3.eth.accounts[int(client)+1]
        if contract.functions.stake(account).call()!=int(round(float(stake)*1e9)):
            raise ValueError('Replay stake mismatch')

def replay(c,output,summary):
    from web3 import Web3
    from .research.evm import EVMAdapter
    output=Path(output);folder=output/'evm-replay'
    binding=replay_binding(c,output,summary)
    if (folder/'agreement.json').exists():
        agreement=json.loads((folder/'agreement.json').read_text())
        verify_cache(agreement,binding,folder)
        deployment=json.loads((folder/'deployment.json').read_text())
        web3=Web3(Web3.HTTPProvider(c.rpc_url))
        artifact=json.loads((Path(__file__).resolve().parents[2]/'blockchain/build/TrustLedgerV4.json').read_text())
        contract=web3.eth.contract(address=deployment['address'],abi=artifact['abi'])
        if agreement.get('contract_address')!=contract.address:raise ValueError('Replay deployment changed')
        verify_live_contract(web3,contract,c,binding)
        return agreement
    if folder.exists() and any(folder.iterdir()):folder.rename(folder.with_name(folder.name+f'.interrupted-{time.time_ns()}'))
    adapter=EVMAdapter(replace(c,experiment_id=binding['experiment_id']),folder)
    trace=json.loads((output/'chain-trace.json').read_text())
    adapter.genesis(trace[0])
    for row in trace[1:]:adapter.round(row)
    verify_live_contract(adapter.web3,adapter.contract,c,binding)
    agreement={'status':'passed','binding':binding,'contract_address':adapter.contract.address,
        'transactions_sha256':digest(folder/'transactions.jsonl'),
        'transaction_count':len(adapter.receipts),'gas_used':sum(r['gas_used'] for r in adapter.receipts)}
    dump(folder/'agreement.json',agreement)
    return agreement
