"""CPU trace profiling and real EVM replays at 20/50/80 clients (development only)."""
import json
from pathlib import Path
from .cli import _execute_config
from .trust_matrix import cell
from .evm import EVMAdapter


def main():
    root=Path(__file__).resolve().parents[2]
    output=root/'results/trust-systems-final'
    rows=[]
    for clients in (20,50,80):
        config=cell('systems',dataset='synthetic',num_clients=clients,
            rounds=5,seed=3,evidence_tier='development',device='cpu',
            synthetic_samples=2000,defense='two_layer',attack='sign_flip')
        directory=output/config.experiment_id
        summary=_execute_config(config,directory,root/'data')
        adapter=EVMAdapter(config,directory/'evm-replay')
        trace=json.loads((directory/'chain-trace.json').read_text())
        adapter.genesis(trace[0])
        for record in trace[1:]: adapter.round(record)
        for client in summary['eligible_client_ids']:
            actual=adapter.contract.functions.stake(adapter.account(client)).call()
            if actual!=adapter.units(summary['final_stakes'][str(client)]):
                raise AssertionError('stake mismatch after full trace replay')
        chain=json.loads((directory/'evm-replay/chain-summary.json').read_text())
        rows.append({'num_clients':clients,'rounds':5,'evidence_tier':'development',
            'runtime_seconds':summary['runtime_seconds_total'],
            'communication_payload_bytes':summary['transmitted_bytes'],
            'ledger_bytes':summary['ledger_bytes'],**chain})
        (output/'systems-summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')

if __name__=='__main__': main()
