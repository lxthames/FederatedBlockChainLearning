# Bug repairs and guarded-recovery candidate v4

19 October 2026. New implementation: `scripts/trust_recovery_v4/`.
New contract: `blockchain/contracts/TrustLedgerV4.sol` and separate compiled
`blockchain/build/TrustLedgerV4.json` (`node compile-v4.mjs` from blockchain/).
Previous sources, bytecode and completed experiments are preserved unchanged.
This is a development candidate; no publication-readiness claim follows.

## Confirmed review issues repaired in this version

1. Paired losses require identical finite nonempty one-dimensional shapes before
   subtraction. Scalar/unequal-length broadcasting now raises an error.
2. An OS-backed exclusive coordinator lock prevents concurrent writers and
   releases on exception/process termination. The lock file may persist; its
   existence alone does not indicate ownership.
3. Replay agreement binds trace, configuration, summary, contract artifact,
   experiment, active model hash/version and final stakes. Reuse verifies the
   binding, success receipts/log digest, live chain ID, deployed experiment,
   active commitment and stakes. Stale caches raise rather than silently pass.
   This detects accidental mismatch; it is not authentication against an actor
   who can replace all local evidence and the trusted node.
4. Runner summaries include the complete package source fingerprint map,
   including loss voting, adaptive attack and recovery guard modules. Study
   freezes additionally include the compiled v4 contract and Solidity source.
5. The v4 contract rejects an under-staked proposer. Candidate creation supplies
   the entire declared eligible snapshot. The contract validates its stake,
   enrollment, uniqueness, proposer exclusion, committee membership and exact
   C=min(E,max(3,ceil(.3E))) size. The registrar still attests completeness and
   off-chain trust eligibility; the contract does not independently know honest
   validation scores or the full off-chain trust registry. Do not overclaim this.

Old code intentionally retains its old semantics and old reproduced issues.
Only the v4 entry point/contract contains these fixes; do not treat previous
results or deployed bytecode as retroactively repaired.

## Recovery design change

Keep rank-loss voting and its screened clipping/trust settings unchanged.
A fixed-panel trend alarm alone no longer authorizes protocol recovery.
Compare each member's current and historical-checkpoint cross-entropy losses on
the SAME private examples. For d_i=loss(current,i)-loss(checkpoint,i), require

    mean(d) > .02 + 2 * sd(d)/sqrt(n), with n>=2.

Recovery requires at least ceil(2C/3) members who BOTH attest local degradation
and approve checkpoint benefit; use the full fixed panel as quorum denominator.
This is a heuristic repeated-data screening margin, not an independent confidence
bound, proof of poisoning, guarantee of future utility, or backdoor detector.
The same validation data is reused; class skew/adaptive selection still matter.

Monetary/trust recovery penalties default to disabled. A separate explicitly
enabled variant tests whether the guard makes sanctions acceptable. This keeps
restoration benefit and punishment separate. Enabling sanctions still applies
only to recorded affirmative signers at most once per recovery event; false
alarms can still punish honest signers and must be reported. Scheduled recovery
controls bypass the benefit guard intentionally to test state transitions;
they are distinct from the end-to-end protocol detector.

## Bounded development study

Run `.venv-trust/Scripts/python.exe -m scripts.trust_recovery_v4.study`.
Status/results: `artifacts/guarded-recovery-v4/` and
`results/guarded-recovery-v4/`. Never launch a second coordinator.

At most 27 new runs: three variants x three development seeds (11/17/23)
x clean/gradual/targeted replacement:

- rank voting plus trust, recovery off;
- guarded recovery, penalties off;
- guarded recovery with explicitly enabled penalties.

Same training-only FEMNIST development data, writer sets, 40 rounds, one epoch,
batch 16, SGD lr .05/momentum .9, 50 clients and .4 participation. No official test
arrays are loaded. Same optimizer, partitions and random streams across variants.
This reuses search-exposed development data; independent replication and transfer
remain necessary. Old 40-round FedAvg attack controls established attack effect
on this training configuration; defend-vs-baseline claims still require a final
matched protocol evaluation and full reported seed groups.

All nine clean cells run first. A complete clean variant must lose <=.05 mean
utility and <=.10 at every seed relative to the new rank-plus-trust baseline,
have <=4 no-update/unsupported rounds per run, and impose ZERO clean monetary
penalties. These are prespecified development screens, not security guarantees.
Failing variants have their attacked cells held. No automatic confirmatory job.
The study records all failed/held cells, restored checkpoints, confirmation
reports, residual ASR, final utility, false alarms, availability and penalties.

Replay one full guarded real-data trace on the separate v4 local contract.
If that trace has no recoveries, it verifies ordinary integration only; synthetic
regressions separately exercise append-only restoration and sanctions. Local EVM
receipts do not establish public-network scalability or economic deterrence.

Detector round-level FPR/FNR semantics are preserved, not relabeled as independent
episode error probabilities. The review's metric interpretation limitation remains.
