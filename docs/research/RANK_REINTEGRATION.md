# Loss-ranking reintegration development study

19 October 2026. This study follows the completed mathematical screening.
Loss ranking passed that exploratory screen with a mean clean utility decrease
of about 4.4 percentage points. Its reduction of targeted ASR did not eliminate
the backdoor. Reintegration remains development work, not publication validation.

Entry point: `scripts/data_preparation/run_rank_reintegration.py`.
Status/analysis: `artifacts/rank-reintegration-v1/`.
New results: `results/rank-reintegration-v1/`.

## Controlled comparisons

Reuse all nine exact rank-only reference runs from `results/trust-math-lab-v1/`
(three seeds x clean/gradual/targeted replacement). Require exact configuration
and configuration-hash agreement, the original source fingerprints, unchanged
development arrays, the completed prerequisite assessment and recorded reference
summary hashes. No old summary is modified.

Add at most 27 development runs:

| Variant | Approval-based trust changes | Recovery | Recovery penalties |
|---|---|---|---|
| Reused rank only | Disabled | Disabled | Disabled |
| Trust only | Enabled | Disabled | Disabled |
| Recovery only | Disabled | Enabled | Disabled |
| Combined | Enabled | Enabled | Enabled |

All keep the screened rank score, retention .8, norm penalty .001, norm clipping
at twice the median, quorum, detector and optimizer settings unchanged. The
combined comparison includes the interaction of restoration and penalties; it
does not isolate penalties alone. Follow-up penalty-only attribution would need
an additional matched variant if the scientific claim requires it.

Use FEMNIST training-only development holdouts, seeds 11/17/23, 50 clients,
40% participation, nominal 30% poisoning, 40 rounds, one epoch, batch 16,
SGD lr .05 and momentum .9. Reuse of the same writer sets and search-selected
method limits generalization. Official test arrays are not read by this study.

Nine clean runs precede attacked runs. For each complete three-seed variant,
require mean clean utility loss relative to rank-only <= .05 and no individual
seed loss > .10. Each run must have <= max(2, .1 * rounds) unsupported/no-update
rounds. Clean honest penalties must be <=5% of initial aggregate stake per run.
These are disclosed engineering screens, not economic or statistical guarantees;
any nonzero wrongful penalties remain reported. Failing clean variants have
their six attacked cells held with explicit reasons. No large confirmation starts.

## Recorded evidence and checks

Analysis keeps seed-level paired utility effects versus rank-only, targeted ASR,
recovery count, FPR/FNR/delay, recovery benefit/regret, honest penalty amounts,
wrongful slashing, quorum failures, accepted fractions and eligibility fractions.
Runtime remains affected by concurrent original classification/SIFT workloads.

The first completed combined real-data trace is replayed in full on the local
Hardhat EVM. Verify final model pointer and all stake balances; the adapter also
checks parent/model state transitions and records receipts. Replay costs and
latency are local measurements. Scheduled synthetic tests exercise recovery even
if the real-data trace has none. A no-recovery trace alone cannot validate the
numerical detector or the recovery path.

Six focused tests passed in 5.50 seconds with TRUST_EVM_TESTS=1: manifest
mechanism isolation, exact reference reuse, clean-gate rejection of missing
seeds/availability collapse/excess honest penalties, restored-checkpoint parent
pointers and preserved economics with/without sanctions, and actual contract
state agreement for a rank-voting recovery trace.

Source, manifest, reference summaries, prerequisite assessment and data hashes
are frozen. Preserve interrupted output directories and resume only with no
existing coordinator. Do not edit the launcher or `scripts/trust_math_lab/`
while this job runs. A systems failure is recorded separately from training.
Source/configuration changes require a new study version and output root.

Next review must assess absolute residual targeted ASR, not only improvements
over an undefended baseline. Any promising integrated variant needs further
development replication, cross-dataset transfer, adaptive/corrupt-validator
tests, complete final-protocol five-seed evidence and the manuscript claim audit.
