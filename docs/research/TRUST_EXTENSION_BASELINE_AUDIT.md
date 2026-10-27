# TRUST extension: baseline and protocol audit

19 October 2026. Research code is isolated under `scripts/trust_extension/`.
The original live `scripts/research/` implementation is preserved. The extension
is a distinct protocol version, `trust-extension-v1`; results cannot be pooled
with v2. Its learning-quality evidence remains pending.

## FedVal

Primary paper: https://www.usenix.org/conference/usenixsecurity23/presentation/valadi

Official code: https://github.com/viktorvaladi/FedVal

Audited author revision: `e5efe74423e067490251948ef6952961dac4f11d`.
Downloaded bytes match that revision. The source hash, paper hash and historical
TFF loader revision are in `artifacts/trust-extension-sources/source-audit.json`.
Author code is retained only in ignored audit artifacts; it is not executed or
vendored as the experiment adapter.

The public code calls `get_points_label` with more arguments than its signature
accepts, calls the parallel evaluation function with a mismatched signature and
references an unset `self.data`. Its adaptive candidate set also differs from
the paper. Therefore using it unchanged would not be a valid executable baseline.

The new implementation is explicitly paper based:

- Evaluate each submitted candidate model on disjoint reserved reference data.
- Use label-conditional cross-entropy losses, with the authors' probability floor
  0.0001. Require reference coverage of every output class.
- For each class, calculate the client mean loss and mean absolute deviation
  across submitted client losses. This follows the operational mean/deviation
  structure of the released code; the paper's notation for the MAD indices is
  ambiguous and is not presented as a uniquely established interpretation.
- Apply Eq. 2 with slope coefficient s1=3 and additive C*s1=9 per label. Use
  class-mean loss divided by mean loss across classes for the bias term.
- Clip total scores at zero and normalize. Equal losses contribute only the
  common baseline, avoiding undefined 0/0 without creating a preference.
- Search s2, s2+0.5, s2-0.5, s2-5, s2+5 each round, initialized at 3, selecting
  the aggregate with the lowest macro-label reference loss. Use the paper's
  label-only score; the released code's additional overall-loss term is omitted.
- Aggregate weighted model deltas with no extra sample-count multiplier.
- Fail explicitly on missing class support or invalid scores.

Tests check hand-computed weights, identical-loss behavior, invalid inputs,
reference class coverage and identical model/vote traces when test labels change.
They verify these operations, not reproduction of the authors' reported accuracy.
Report this implementation choice in the paper, rather than claiming bitwise
agreement with the non-executable released implementation. No reference data is
used to tune the proposed framework's votes or monitor.

## Recovery and adaptive attacks

The extension adds one material protocol guard: each monitoring member has its
own detector history over the fixed window, and at least ceil(2C/3) local
degradation attestations are required in addition to the median-quality trigger.
The EVM snapshot remains the full panel; only actual affirmative attesters send
recovery attestations. Shrinking the panel to the affirmative subset is forbidden.
This guard must be validated by the extension's own completed evidence.

False-recovery actors publish a prespecified declining quality sequence; vote
withholders abstain. Honest validators use only their private validation data.
The simulator's corruption labels inject adversarial behavior and score errors;
they are not inputs to the honest decision rule. Quorum does not prove numerical
truth, and colluding quorums remain a failure case to measure.

Scheduled controls suppress automatic recovery before round 20 and compare
continuing training, restoration without sanctions and restoration with sanctions
at the same checkpoint. Their trace histories must agree before intervention.
These controls do not implement FedRecover, Crab, FLAME or FLDetector. Those
methods provide literature context and should not appear as measured baselines.

The adaptive attack is a bounded proxy-data constrain-and-scale search. Its
clean-proxy tolerance is 0.02 absolute and its norm limit is 1.5 times the benign
local delta. Eight prespecified scales are evaluated, including a feasible zero
update. It uses attacker-owned training examples, with a 256-example cap,
and public weights. It has no access to committee validation or held-out tests.
FEMNIST's inverse-background representation requires a black (zero) trigger;
Fashion-MNIST uses a white (one) trigger. This distinction is tested.

The original v2 ALIE uses fixed z=1, omniscient honest-update references and
fails when fewer than two are available. It remains labeled as a variant;
insufficient references are recorded failures. Original SIFT FLTrust cells with
no positive root alignment also remain explicit failures. Neither condition is
silently replaced with another algorithm or counted as a successful defense.

## Dataset integrity

The full TFF FEMNIST archive SHA256 is
`fe1ed5a502cea3a952eb105920bff8cffb32836b5173cb18a57a32c3606f3ea0`,
matching the official loader at revision
`a484b84e053dca1a1d6afa29b86a3cd5ece9dd9d`.

Select 50 enrolled writers per seed using IDs and minimum sample counts, then
reserve different writers for 10 reference examples per class. Keep official
test samples. Remove training/reference pixel duplicates, including matches to
tests, without consulting test labels; then split enrolled training data 80/20
into training/private validation. Verify complete index assignment, disjoint
writer roles, all file hashes and zero exact-pixel overlap with tests. Test-set
pixel duplicates are retained and their unique count is reported. Do not call
test rows independent statistical replicates; uncertainty uses seed-level runs.

The raw sample availability filter is fixed before inspecting utilities. If
deduplication leaves insufficient data or class support, preparation fails.
Worst-client results include sample counts because very small test shards are
noisy. This is a natural-writer benchmark, not evidence of demographic fairness.
