# Active experiment code review — 19 October 2026

Scope: mathematical voting, active rank reintegration coordinator, runner,
checkpoint ledger, EVM adapter and TrustLedger contract. This is a bounded
correctness review, not a complete security audit of the repository. No frozen
training source, contract artifact or experiment result was changed.

## Confirmed findings

### Medium: cached contract agreement bypasses validation of the replay trace

`scripts/data_preparation/run_rank_reintegration.py:118` returns whenever
`evm-replay/agreement.json` exists. It does not verify the saved status, trace
hash, deployment identity or expected final state. A resumed run with a modified
or mismatched trace can reuse stale agreement. Reproduced by creating a changed
trace and agreement with an incorrect trace hash; replay returned successfully.
This does not show a fresh replay was wrong. It prevents treating the cache as
verified evidence without independent checking.

Fix in a separately versioned coordinator: verify trace digest and identity,
expected model pointer/stakes and agreement status; reject mismatched cache.
Hash the completed trace together with its summary in the output manifest.

### Medium: contract boundary does not enforce all simulator eligibility rules

`blockchain/contracts/TrustLedger.sol:78-88` verifies validator enrollment/stake,
proposer exclusion, unique members and a minimum committee size. It does not
require sufficient proposer stake or the specified committee-size formula.
On newly deployed existing bytecode, an enrolled zero-stake proposer opened a
candidate with four staked validators. E=4 calls for C=3, but C=4 was accepted.
This is a genuine enforcement gap relative to simulator behavior. It is not a
demonstrated attack by an unprivileged user: only the trusted registrar can call
openCandidate. The contract also has no on-chain trust-score registry, so the
full eligible population cannot currently be independently reconstructed there.

Fix proposer stake enforcement in a versioned contract. For committee size,
either enforce a committed eligibility snapshot and its policy, or explicitly
scope size/trust eligibility to the trusted orchestrator and test that boundary.
Do not claim the existing contract independently verifies all eligibility rules.

### Low: paired-loss input lengths silently broadcast

`scripts/trust_math_lab/loss_voting.py:13` subtracts arrays before checking equal
shapes. Input baseline=[1,2,3], candidate=[.5] produces three differences and an
approval, instead of rejecting invalid pairing. Reproduced with actual helper.
Current loss committees use identical indices; current rank-only/reintegration
runs do not use this paired-decision helper. The defect is an input-validation
gap rather than evidence those ranking outcomes are corrupted.

Fix: require both inputs to be finite, nonempty one-dimensional arrays with
exactly equal shapes before subtraction; test unequal lengths and scalars.

## Other limitations identified by source inspection

- No exclusive coordinator lock: starting the same entry point twice can race
  status/summary writes and interrupted-directory renames. Manual instructions
  currently prevent duplicate launch; implement an OS-backed process lock in a
  versioned coordinator before unattended orchestration. This race was not
  deliberately exercised against live results.
- Runner summary source_hash covers only `research/*.py`, excluding the parent
  loss_voting/fedval/adaptive modules. Study-level frozen source maps include
  these modules and compensate, but a standalone summary cannot identify the
  full implementation. Preserve outer frozen manifests with summaries.
- Detection FNR counts attacked non-flagged rounds, including detector warm-up,
  not independent missed attack episodes. Round-level FPR/FNR are described as
  attack-presence metrics in the manuscript; report observation opportunities
  and episode/delay measures too. Do not interpret that FNR as a calibrated
  probability that a poisoning event escapes detection.

## Reproduction evidence and impact on running work

`scripts/data_preparation/audit_active_protocol.py` writes
`artifacts/code-review-v1/reproductions.json` and transaction receipts under
`contract-reproduction/`. Only a fresh local audit contract was deployed; no
existing study contract was modified. The script intentionally reproduces flaws
and is not an implementation fix or proof of their absence elsewhere.

The active rank study can continue as exploratory learning evidence: none of
the reproduced issues establishes an error in the ongoing rank numerical path.
Its cached replay evidence requires a separate trace/state check before reuse,
and its on-chain enforcement claims need the explicit boundary above. Do not
mutate frozen source while it runs. Implement repairs with a new coordinator/
contract version and regressions, then verify artifact compatibility before any
publication use. No claim that the code is bug-free is justified.

## Repair follow-up

The separately versioned v4 implementation repairs paired shapes, cached replay
validation, coordinator locking and full-package source fingerprints. Its new
Solidity artifact enforces proposer stake and committee size/membership against
the registrar's explicit eligibility snapshot. Snapshot completeness and trust
eligibility remain trusted-orchestrator assumptions. Original artifacts were
preserved. See GUARDED_RECOVERY_V4.md for exact boundaries and development checks.
