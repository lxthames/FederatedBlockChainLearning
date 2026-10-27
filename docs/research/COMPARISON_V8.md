# V8 bounded comparison and candidate protocol review

Prepared before execution, 19 October 2026. Maximum 44 new GPU runs; reuse 19
completed Fashion-MNIST rows. No automatic confirmatory training follows.

## Candidate review

The candidate is precisely V6 `rank_trust`: private rank-loss votes, norm clip
ratio 2, keep fraction 0.8, norm-score penalty 0.001, trust reward 0.02 and
rejection penalty 0.02, unchanged stake/eligibility and proposer-excluding
committees. `ablation=validation_only` disables recovery. Financial poisoning
penalties are disabled. This is a candidate validation protocol, not an
experimentally established complete accountable-recovery system. Historical
checkpoint recovery remains a separately evaluated mechanism with negative and
conditional development findings. Final-paper equations and claims must reflect
this distinction before any final protocol freeze.

The current implementation is preserved under `scripts/trust_literature_v6/`.
V7 and V6 bindings must verify before any reused results are read. V8 changes
only experiment configuration and coordination, not the candidate algorithm.

## Finite experiment manifest

Use development seeds 11, 17 and 23; 50 enrolled clients, 40% participation,
30% poisoning, one local epoch, bounded/matched participant sampling. Conditions
are clean, gradual poisoning, and targeted model replacement, with onset 11.

Fashion-MNIST: retain V6/V7 settings (30 rounds, Dirichlet alpha 0.5, SGD 0.01,
momentum 0.9, batch 64). Reuse all 18 completed FedAvg/rank controls and the
seed-11 clean median control. Execute the remaining 26 median, FLTrust and
audited paper-based FedVal cells. Baseline fidelity and reference-data assumptions
are recorded in `TRUST_EXTENSION_BASELINE_AUDIT.md` and its pinned source audit.
FedVal is not claimed equivalent to executable author code. FLTrust uses cosine
alignment, root-norm rescaling and normalized nonnegative weights; no positive
alignment is a recorded failure with no fallback. Median uses the existing
coordinate-wise PyTorch median (lower median for an even count).

FEMNIST: 18 cells, FedAvg versus rank/trust, three conditions and three seeds.
Use natural writer splits and disjoint training-only holdouts, 62 classes,
40 rounds, SGD 0.05, momentum 0.9, batch 16. These learning settings were selected
in earlier development work. Do not pool raw accuracy across the two datasets or
call this a matched-hyperparameter cross-dataset comparison. Reference writers
remain disjoint from enrolled writers. The prepared development labels are not
the official test labels. No new dataset downloads or relabeling are involved.

## Gates and interpretation

For new FEMNIST rank/trust attack cells, preserve the V6 per-seed clean gate:
FedAvg utility at least 0.20, utility difference at least -0.05, at most three
unsupported rounds, and zero honest monetary penalties. Missing or failed clean
controls hold that seed's rank attack cells. These requirements are screening
criteria; passing them does not certify fairness or security.

Baseline attack cells run even if their clean utility is poor: this new,
prespecified study measures their robustness/utility tradeoff. Report the clean
gate result alongside every comparator. This does not release or reinterpret
the previously held V6 FLAME, FLShield or detector attack cells. Failure of a
published method in this implementation/settings is not a general refutation.

Report complete three-seed groups only for means, individual seeds and missing
cells, paired participants/data, effective-attack checks, honest exclusion,
runtime and mean targeted ASR. Attack validity thresholds remain one point
gradual damage and ten points targeted ASR above the clean trigger control.
No method is promoted automatically; no post-hoc tuning is permitted here.
Reference access and total validation work differ across methods; equal
training and participant budgets do not imply equal validation budgets.

Remaining baseline gaps include matched Multi-Krum/BOBA results for the selected
candidate and their source/assumption checks. CIFAR-10, 100-round confirmatory
groups, independent validator corruption, drift, detector sensitivities,
final-version live/replay EVM checks, and manuscript reconciliation remain
required. This batch does not resolve all requirements in the original plan.

## Integrity and operation

Freeze all imported candidate/coordinator sources, this policy, the baseline
audit, complete configurations, provenance/data files and reused artifact
digests. Verify data before each new cell. Record software/GPU, OS-exclusive
coordinator lock, structured events and errors. Preserve interrupted folders;
rerun only an incomplete cell. All seeds are search-exposed development seeds
with overlapping training pools. No publication readiness or statistical
confirmation claim follows from descriptive three-seed means.

Launch from repository root:
`.venv-trust/Scripts/python.exe -m scripts.data_preparation.run_comparison_v8`.
Use a hidden background process, redirected stdout/stderr. Status and analysis
are under `artifacts/comparison-v8/`; results under `results/comparison-v8/`.
