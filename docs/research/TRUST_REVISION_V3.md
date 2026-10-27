# Candidate revision v3: development, not a validated final protocol

19 October 2026. Source: `scripts/trust_revision_v3/`. This isolated snapshot
preserves original research and extension sources and results. Only run the new
`development` entry point; copied legacy orchestration modules are not the v3
experiment schedule. The 275-cell follow-up remains stopped.

## Implemented corrections and candidate design changes

- All-rejected committees now produce a recorded no-op and update proposer trust;
  previously the runner skipped the defense, leaving rejection outcomes unapplied.
- An approved zero-trust proposer receives a positive probation aggregation
  weight. The weight is n_k * max(trust_k, 0.1), not a silent FedAvg fallback.
- Low trust prevents committee eligibility but no longer permanently prevents
  proposing. Approval can rebuild trust. Stake below the minimum prevents both
  committee eligibility and subsequent proposals.
- Rejection reduces trust by 0.02; approval adds 0.02. Recovery penalties remain
  0.15 trust and five stake credits. This separates validation disagreement from
  responsibility attributed by a recovery event. All values are configurable.
- Validators compare an absolute utility change against -0.02. They evaluate
  the full candidate on the same private examples as the baseline. Near-zero
  baseline accuracy no longer magnifies tiny absolute changes into large relative
  drops. Relative deltas remain diagnostic only. This is a candidate rule and
  may trade detection power for lower clean rejection; experiments must measure it.
- Preserve the extension's fixed-panel local-attestation recovery quorum.
  Do not merge v2/extension/v3 outcome groups.

Implementation checks: 93 tests passed in 18.29 seconds with TRUST_EVM_TESTS=1.
New tests cover absolute-rule boundaries and attacker-label independence,
all-rejected trust updates, zero-trust probation recovery, disjoint development
evaluation, independence from evaluation labels, stake-depletion eligibility,
and actual EVM recovery/state agreement. These tests do not prove robustness.

## Bounded development manifest

Run: `.venv-trust/Scripts/python.exe -m scripts.trust_revision_v3.development`

Status: `artifacts/trust-revision-v3-development/status.json`.
Analysis: `artifacts/trust-revision-v3-development/analysis.json`.
Results: `results/trust-revision-v3-development/`.

36 cells = two datasets x three new random seeds (11/17/23) x FedAvg/full
x clean/gradual/targeted replacement. All 12 clean cells run first. This is
development evaluation; no confirmatory runs launch automatically.

Both methods use the same splits, optimizer and local training budget:

- Fashion-MNIST: 40 rounds, one epoch, batch 64, SGD learning rate .01, no momentum.
- FEMNIST: 100 rounds, one epoch, batch 16, SGD learning rate .05, momentum .9.
  Momentum is reset with the per-client local optimizer each round. This is a
  prespecified development candidate, not a claim that convergence is established.
- Both: 50 clients, 40% controlled participation, nominal poisoning fraction .3.

The data loader reads only training pools. Fashion-MNIST training images are
deduplicated by exact pixels, then stratified 20% development evaluation is
reserved before the runner creates private training/validation/reference roles.
FEMNIST retains the existing development writer sets (source seeds 3/5/7), their
private validation and separate-writer references; 20% of each client's training
partition is reserved for independent development evaluation and indices remapped.
New random seeds are not new independent writer samples. Source hashes, indices,
arrays and the full manifest are retained. Official test arrays are never loaded
by this entry point. Data reuse validates all stored file fingerprints.

The earlier diagnostic inspected official-test outcomes. That exposure cannot be
undone: disclose it as exploratory evaluation, and do not describe those official
test sets as entirely untouched by development. New training-only diagnostics
prevent additional test-based tuning; final claims require an explicit evaluation
design and this limitation, not simply renaming an old result confirmatory.

Development analysis requires all three seeds. Heuristic clean-learning screen:
each FedAvg run exceeds its development majority-class accuracy by five percentage
points. Clean-cost screen: full-minus-FedAvg mean >= -5 points and every seed >=
-10 points. These are feasibility screens, not statistical guarantees or optimal
threshold claims. Attack gates retain paired utility loss > .005 or targeted-ASR
lift > .05, conditional on useful clean learning. Failed and incomplete groups
remain visible. Further attacks (ALIE, sign flip, adaptive) and final-protocol core
coverage remain necessary before publication.

## Resume and evidence policy

Never start a second coordinator while the recorded worker is alive. Completed
cells are reused only with matching frozen source/data/configuration. Interrupted
cells are moved to a timestamped sibling and rerun. No previous v2/extension result
is overwritten. Do not edit the frozen v3 Python sources during execution.
Do not update manuscript equations to claim v3 is final until development and
attack checks have been reviewed. See READINESS_VALIDATION.md for remaining gates.
