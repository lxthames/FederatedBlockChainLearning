"""Real local-EVM transactions; receipt-backed cost measurements only."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from web3 import Web3
from .config import ExperimentConfig


class EVMAdapter:
    def __init__(self, config, output_dir):
        self.config = config
        self.output = Path(output_dir)
        self.output.mkdir(parents=True,exist_ok=True)
        self.web3 = Web3(Web3.HTTPProvider(config.rpc_url))
        if not self.web3.is_connected() or self.web3.eth.chain_id != 31337:
            raise ValueError('only a connected local Hardhat chain 31337 is supported')
        self.accounts = self.web3.eth.accounts
        if len(self.accounts) < config.num_clients+1:
            raise ValueError('local node needs at least num_clients+1 unlocked test accounts')
        self.sender = self.accounts[0]
        self.receipts = []
        self.receipt_path = self.output/'transactions.jsonl'
        if self.receipt_path.exists():
            raise FileExistsError('transaction log already exists')
        root = Path(__file__).resolve().parents[2]
        artifact = json.loads((root/'blockchain/build/TrustLedger.json').read_text(encoding='utf-8'))
        factory = self.web3.eth.contract(abi=artifact['abi'],bytecode=artifact['evm']['bytecode']['object'])
        identity = hashlib.sha256(config.experiment_id.encode()).digest()
        receipt = self.send(factory.constructor(identity,self.units(config.minimum_stake),self.units(config.slashing_penalty)),self.sender,'deploy')
        self.contract = self.web3.eth.contract(address=receipt['contractAddress'],abi=artifact['abi'])
        deployment = {'address':self.contract.address,'chain_id':31337,'rpc_url':config.rpc_url,
                      'compiler':artifact['compiler'],'experiment_id':config.experiment_id,
                      'stake_unit_wei':10**9,'trust_assumption':'registrar and off-chain aggregation trusted'}
        (self.output/'deployment.json').write_text(json.dumps(deployment,indent=2),encoding='utf-8')

    @staticmethod
    def units(amount):
        return int(round(float(amount)*10**9))

    def account(self, client):
        return self.accounts[int(client)+1]

    def send(self, function, sender, operation, value=0):
        started = time.perf_counter()
        tx_hash = function.transact({'from':sender,'value':value})
        receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash,timeout=120)
        elapsed = time.perf_counter()-started
        transaction = self.web3.eth.get_transaction(tx_hash)
        record = {'operation':operation,'transaction_hash':tx_hash.hex(),
                  'gas_used':receipt['gasUsed'],'input_bytes':len(transaction['input']),
                  'latency_seconds':elapsed,'status':receipt['status'],
                  'block_number':receipt['blockNumber'],
                  'receipt':json.loads(self.web3.to_json(receipt))}
        with self.receipt_path.open('a',encoding='utf-8') as out:
            out.write(json.dumps(record)+'\n')
        self.receipts.append(record)
        if receipt['status'] != 1:
            raise RuntimeError(f'reverted transaction: {tx_hash.hex()}')
        return receipt

    def genesis(self, record):
        for client in record['enrolled']:
            account = self.account(client)
            self.send(self.contract.functions.enroll(account),self.sender,'enroll')
            self.send(self.contract.functions.deposit(),account,'deposit',self.units(self.config.initial_stake))
        self.send(self.contract.functions.genesis(bytes.fromhex(record['model_hash'])),self.sender,'genesis')
        self.save_summary()

    def round(self, record):
        rnd = record['round']
        assert self.contract.functions.activeVersion().call() == record['parent_version']
        accepted = []
        for candidate in record['candidates']:
            proposer = self.account(candidate['client_id'])
            digest = bytes.fromhex(candidate['candidate_hash'])
            committee = [self.account(i) for i in candidate['committee']]
            key = self.contract.functions.candidateKey(rnd,record['parent_version'],proposer,digest).call()
            self.send(self.contract.functions.openCandidate(rnd,proposer,digest,committee),self.sender,'open_candidate')
            for vote in candidate['votes']:
                self.send(self.contract.functions.vote(key,vote['approve']),self.account(vote['validator_id']),'vote')
            chain_accepted = self.contract.functions.accepted(key).call()
            if chain_accepted != candidate['accepted']:
                raise AssertionError('simulator/contract candidate decision mismatch')
            if chain_accepted: accepted.append(key)
        self.send(self.contract.functions.commitModel(rnd,bytes.fromhex(record['model_hash']),accepted),self.sender,'commit_model')
        recovery = record.get('recovery')
        if recovery:
            key = hashlib.sha256(f'{self.config.experiment_id}:recovery:{rnd}'.encode()).digest()
            reason = hashlib.sha256(recovery['trigger'].encode()).digest()
            panel = [self.account(i) for i in recovery['panel']]
            self.send(self.contract.functions.openRecovery(key,recovery['from_round'],rnd,
                recovery['restored_version'],reason,panel,recovery['penalties_enabled']),self.sender,'open_recovery')
            for validator in panel:
                self.send(self.contract.functions.attestRecovery(key),validator,'attest_recovery')
            self.send(self.contract.functions.applyRecovery(key),self.sender,'apply_recovery')
        expected = recovery['restored_version'] if recovery else rnd
        if self.contract.functions.activeVersion().call() != expected:
            raise AssertionError('active model pointer mismatch')
        self.save_summary()

    def save_summary(self):
        summary = {'transaction_count':len(self.receipts),
                   'gas_used':sum(x['gas_used'] for x in self.receipts),
                   'input_bytes':sum(x['input_bytes'] for x in self.receipts),
                   'latency_seconds':sum(x['latency_seconds'] for x in self.receipts),
                   'active_version':self.contract.functions.activeVersion().call(),
                   'local_only':True}
        (self.output/'chain-summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--trace',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    config = ExperimentConfig.from_file(args.config)
    adapter = EVMAdapter(config,args.output)
    records = json.loads(Path(args.trace).read_text(encoding='utf-8'))
    adapter.genesis(records[0])
    for record in records[1:]: adapter.round(record)


if __name__ == '__main__': main()
