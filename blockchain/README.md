# Local EVM implementation

Requires Node.js 24 and npm. Install locked packages with `npm.cmd ci`, compile
with `npm.cmd run compile`, and start the local node with `npm.cmd run node`.
The source contract is `contracts/TrustLedger.sol`; `compile.mjs` records the
complete compiler input with the ABI and bytecode under `build/TrustLedger.json`.
Hardhat is pinned in package-lock.json. The node uses 100 development accounts,
binds to localhost, and exposes chain ID 31337.

Python runs deploy independent contract instances. Addresses and transaction
receipts are saved inside each experiment directory. The registrar selects
committee snapshots; authentication and vote uniqueness are enforced on-chain.
Model arithmetic and validation reports remain trusted off-chain computations.
Recovery requires a panel-attestation quorum and preserves all commitments.

No withdrawal interface is implemented in this research prototype. Deposits and
penalties use local test funds. Contract gas and local latency do not establish
production consensus performance or an economic deterrence result.

From the repository root, with the node running:

```powershell
$env:TRUST_EVM_TESTS='1'
.\.venv-trust\Scripts\python.exe -m pytest tests/test_evm_integration.py -q
.\.venv-trust\Scripts\python.exe -m scripts.research.systems_check
```

The integration test covers unauthorized enrollment, duplicate committee members,
proposer exclusion, unsupported committees, unauthorized/duplicate/stale votes,
missing quorum, append-only recovery, and idempotent penalties. The systems check
replays complete five-round development traces for 20, 50 and 80 clients, saving
measured receipts and checking final simulator/contract stake agreement.
