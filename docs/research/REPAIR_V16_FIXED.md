# V16 fixed aggregate validation and counterfactual repair

## Integrity restart after V15

V16 fixed repeats the V15 algorithm and finite matrix with stronger execution checks.
Random-stream domain strings intentionally retain their V15 names for exact parity.
No thresholds, score, gate, architecture, training budget or repair policy changed.
All V16 fixed cells and calibration are rerun; old V15 results are not reused or overwritten.

The V15 Fashion-MNIST aggregate seed-11 model committed at round 9 differs from
replay in exactly three of 401,408 entries in layer 7.weight. Values near zero
became approximately 143.806, 1457.928 and -128.216. All other model entries match.
The archived round-11 parent has the recorded corrupted round-9 model hash.
An instrumented replay and an ordinary replay agree on the uncorrupted trajectory.
This invalidates interpreting its 40.70% endpoint/91 holds as an algorithmic failure.
Underlying hardware/runtime cause remains unknown; a source-code cause is not proved.
The old worker was stopped with artifacts preserved. See the separate audit memo.

Every application of a delta now compares the actual result bitwise with an
independently computed CPU parent-plus-delta reference. The runner verifies the
parent at round start and before/after validation, verifies committed weights,
binds commitment to the actual model scored by the panels, and checks state after
evaluation. Behavioral inference also verifies that parameters stayed unchanged.
Mismatches stop the cell/queue and preserve error evidence; no silent correction
or automatic retry. These checks cannot rule out all hardware faults or corruption
of training updates before a trusted reference is established.

Each cell runs in a fresh hidden Python subprocess to isolate process state and
release resources. This is a precaution, not evidence of the cause or a guaranteed
repair of an underlying device problem. Integrity work is included in runtime;
compare measured overhead within V16 fixed, not directly against uninstrumented V15.
Every completed cell seals integrity-summary.json and per-round model_integrity
events. Historical V15 sources and test hashes must still match their original freeze.

Prepared October 20, 2026 (Taipei), before execution. Long-horizon development
on exposed seeds 11/17/23; not independent confirmation. Novelty, Byzantine
robustness, fairness and publication readiness are unestablished. Preserve V13/V14
sources and results. New source: scripts/trust_repair_v16_fixed; results: results/repair-v16-fixed;
status: artifacts/repair-v16-fixed. No tuning or automatic confirmation follows this batch.

## Frozen matrix

Maximum 114 cells of 100 rounds: six passive clean-clipping calibration runs,
36 clean evaluation runs, then up to 72 attack runs. Two datasets (Fashion-MNIST,
natural-writer FEMNIST), three exposed seeds, six arms (FedAvg, clipping,
paper-based FedVal adapter, current scaled-loss quorum, aggregate gate, repair),
three conditions (clean, bounded gradual reversed updates, existing targeted model
replacement). Fifty clients, 20 participants/round, wide CNN, one epoch, inherited
SGD settings, attack onset 11, 30% bounded attacker roles. Matched initialization,
data, client partitions, fixed evaluation groups and participant schedules.
Do not pool endpoints with shorter V13/V14 runs. No new attack family or sweep.
FedVal retains central class-complete reference access and a different validation
budget; it is the audited adapter, not a certified author-code reproduction.

## New algorithm

Gate and repair start from clipping: include all finite clipped proposals,
sample-weight and renormalize to one, fixed trust, no rank rejection or penalties.
The inherited keep-all path still computes local rank votes; count that overhead.
Validate the exact normalized aggregate, not individual nominal sample-share probes.

Each validator samples at most eight of its own private examples with independent
deterministic streams. Challenge bank: clean plus 7x7 spatial grids of black 2x2,
white 2x2 and gray 4x4 patches, plus Gaussian noise SD .08 clipped to [0,1].
For 28x28 images there are 149 blocks. Grids include all boundaries but are not
exhaustive. Sizes/colors/corners overlap the current attack family: no unseen-trigger
claim. Banks never read attack settings, target labels, identities, test data or ASR.
Transformations may alter legitimate predictions; no transformed labels are assumed.

For every transform/class compute mean transformed probability minus mean clean
probability. Absolute score is the maximum positive contrast; incremental score is
the maximum positive increase of that contrast relative to the parent. Preserve
full parent/candidate transformation-class contrasts as compressed NPZ telemetry.
Also compute paired clean cross-entropy: clean_excess=max(0,mean_change-2*SE).
Absolute response is not proof of a backdoor; incremental response can miss a
backdoor inherited from a compromised parent. These are heuristic measurements.

## Calibration and panels

Six 100-round clean clipping trajectories supply passive private-data measurements.
Freeze the higher empirical 99.5th percentile of each metric by dataset and panel
stage; bind all six completion seals. The maximum-over-transform operation occurs
before quantile calculation. Later clean clipping must reproduce every calibration
model hash. No attack or evaluation outcomes select thresholds. Calibration/evaluation
reuse exposed seeds, and correlated rounds/votes confer no independent FPR guarantee.
Clean clipping trajectories may not match the deployed defense distribution.

Sample ten eligible noncontributors per round without replacement: five screening,
five verification, disjoint. A panel passes if four of five validators satisfy all
three metric thresholds. This is distinct from the current candidate's 10-of-15
quorum. Validators are honest here. The coordinator simulates private validation;
there is no cryptographic privacy or Byzantine-safety claim, and scores may leak data.

## Bounded repair

Both full-aggregate panels must pass. Aggregate-only otherwise holds the round.
Repair searches only when screening fails. A verification-only failure holds without
searching against that feedback. Test every single-client omission on screening,
renormalizing remaining sample weights. Require screening success and behavioral
risk improvement over the full aggregate by more than 1e-6. Validator risk is
max(absolute/max(threshold,1e-6),incremental/max(threshold,1e-6)); panel risk is its
median. Select the lowest-risk eligible omission; client ID breaks exact ties.
Verify only the selected omission, once. If verification fails or none improves,
hold the round. No retries using verification feedback. Reused verification data
and rounds still do not supply a statistical generalization guarantee.

At most one omission per round. Colluding attackers can defeat this restriction.
Omission is measured marginal contribution, not malicious intent; no monetary
penalty. Holds count as unsupported, all omitted updates count as excluded.
Distinguish candidate approvals from final inclusion. At most 23 aggregate panel
evaluations per failed round: full screen/verify, 20 omission screens and selected
verification. Cache parent predictions per validator. Count validation work, model
delivery estimates, runtime, repairs, holds and storage.

Archive parent/proposed delta at rounds 1/11/51/100. In repair cells also archive
parent/proposed delta/all clipped updates at the first three repair-triggering rounds.
Seal all telemetry/checkpoints plus config, summary, splits, events, trace, external
settings and diagnostic bindings. These are partial diagnostics, not full recovery
history. Preserve interrupted directories; resume verifies completion seals.

## Gates and reporting

Run every clean cell. Current/gate/repair attacks require all six clean cells for
that arm to pass: strict matching to clipping, clipping accuracy >=20%, <=5 points
utility deficit, <=3 unsupported rounds, zero honest monetary penalty. Controls
always run. Held attacks are recorded with reasons and are not defense successes.
No gate relaxation or favorable-seed selection. Integrity violations/crashes stop
the queue for inspection. Means require all three matched seeds, and clean/attack
models must match through round ten. Do not average partial or mismatched groups.

Report seed-level ordinary and lower-tail utility, mean and endpoint ASR, clean
trigger prediction rates, honest exclusion, malicious final inclusion, repair/veto
counts, validation/communication work, storage and runtime. Interpret targeted
benefit only if matched-control mean ASR exceeds its clean trigger rate by ten
points; retain utility and compare clipping/relevant baselines, not just FedAvg.
Inspect attack trajectories for general target collapse. There is no automatic
novelty, significance, defense-advantage or publication-readiness declaration.

## Execution

Run tests/test_repair_v16.py before freeze. Then module
scripts.trust_repair_v16_fixed.study --freeze, followed by --run with CUDA in a hidden
Windows background process and redirected stdout/stderr. Monitor status.json and
worker logs under artifacts/repair-v16-fixed; active cell events.jsonl gives round
progress. comparison.md and analysis.json update after completed/held cells.
Fresh independent calibration/seeds, unseen/adaptive attacks and a primary-source
novelty/fidelity audit remain prerequisites for a general defense claim.
