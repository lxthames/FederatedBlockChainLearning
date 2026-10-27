# V9 validation-bias mechanism pilot — protocol before training

19 October 2026. Maximum 42 new runs, including two mandatory reproduction
checks. No automatic confirmation, tuning, monetary penalties, or recovery.
Implementation is isolated under `scripts/trust_bias_v9/`; V6–V8 sources and
completed results remain unchanged. All candidate training is GPU background work.

## Diagnosis and mathematical motivation

The completed V8 comparison shows FEMNIST clean damage and stronger Fashion-MNIST
performance from the audited paper-based FedVal comparator. The descriptive
audit `artifacts/bias-v9-pilot/diagnosis.json` uses only clean training/private
validation labels. Approval correlates with proposer/validator label-histogram
overlap. Votes share models, rounds and examples: neither significance nor
causation follows from treating these votes as separate observations.

For validator v with local distribution P_v, its score estimates a change in
L_v, not in global L = sum_j p_j L_j. In general grad(L_v) dot delta_i can be
positive (local loss increases) while grad(L) dot delta_i is negative (global
loss decreases). More concretely,
L_v(w + delta_i) - L_v(w) is approximately grad(L_v) dot delta_i, while the
round applies only a sample-weighted share of each update. Evaluating the full
local step exaggerates a single client's contribution and nonlinear effects.
Further, keeping ceil(0.8m) proposals ranks and rejects a tail even when every
submitted update is benign and useful. A larger voting quorum does not turn
biased local objectives into an unbiased global objective.

Zeno supplies primary-source context for loss-decrease/norm-penalty ranking:
https://proceedings.mlr.press/v97/xie19b.html . Its theorem is not transferred to
private heterogeneous validation or the following adaptations. Heterogeneity
and robust aggregation are also treated by Fixing by Mixing:
https://proceedings.mlr.press/v206/allouah23a.html . NNM is not implemented in this
pilot; mixing original proposals would introduce a separate provenance and
penalty-attribution question. This pilot isolates validation behavior.

## Fixed variants

Let delta_i be the clipped submitted update, a_i = n_i / sum_j n_j, and M the
median clipped-update norm. No simulator attacker label enters these rules.
The actual aggregator and all committee/eligibility rules remain unchanged.
The probe scale is nominal sample share, not the final accepted/trust-weighted
share: no exact marginal-contribution or Shapley-value claim is made.

- `probe`: score local mean [loss(w)-loss(w+a_i delta_i)] minus
  0.001 (a_i ||delta_i|| / M)^2. Rank the best 80%, including cutoff ties.
  Actual accepted updates retain their full submitted values during aggregation.
- `macro_probe`: same probe and norm score; compute loss improvement as the
  unweighted mean of per-class means on classes observed by that validator.
  Missing classes remain unknown, with no imputation. This corrects local class
  frequency weighting, not cross-writer coverage bias or global unbiasedness.
- `margin_probe`: same ranking as `probe`, but also approve a rank-rejected
  proposal when its paired probe loss increase is at most 2 times its sample
  standard error (zero additive tolerance, at least two examples). This is a
  fixed heuristic backoff, not a simultaneous confidence guarantee or equivalence
  test. Correlation, repeated evaluations and data reuse invalidate a naive
  significance interpretation. It may also admit harmful attacks: measure them.
- `clip_only`: rank keep fraction 1, fixed trust (zero reward/rejection penalty),
  clip ratio 2. All finite proposals tie/pass the minimum cutoff, so this is a
  sample-weighted clipping control, with committee records retained. It tests
  whether validation is needed beyond clipping. It is not a security claim.

## Manifest and gates

First reproduce the unchanged original rank rule on clean Fashion-MNIST seed 11
and FEMNIST seed 11 at the previous horizons. Require identical model hashes
at every recorded round, participant sequences, data fingerprints and final
utility. Any parity failure stops the entire study.

Then execute four variants on FEMNIST clean seeds 11/17/23 and Fashion-MNIST clean
seed 11 (16 cells). FEMNIST settings remain 40 rounds, SGD 0.05/momentum 0.9,
batch 16, 50 natural writers, 40% participation. Fashion settings remain
30 rounds, SGD 0.01/momentum 0.9, batch 64, alpha 0.5. All have 30% potential
attacker identities, independently sampled validator corruption 0, and attack
onset 11. Prepared training-only development holdouts are used throughout.

A variant proceeds to its six FEMNIST gradual/targeted cells only if all three
FEMNIST clean seeds AND Fashion clean seed 11 pass the unchanged V6 per-seed gate:
FedAvg utility >=0.20, candidate difference >=-0.05, unsupported rounds <=3,
and zero honest monetary penalty. Hold all six attacks if any required clean
cell fails/is missing. No threshold relaxation or fallback methods.

Compare each variant to matching FedAvg and original rank controls. Report every
seed, gate outcome, failed/held cell, class support, benign-update exclusion,
honest-role exclusion, runtime and mean post-onset targeted ASR. Only complete
three-seed groups may supply means. Attack effectiveness against FedAvg remains
at least one utility point gradual damage and ten ASR points above clean trigger
rate for targeted attacks. Review requires mean improvement versus FedAvg of
at least one utility point (gradual) or five ASR points (targeted), no seed's
attack utility loss greater than five points, and matched participants/data.
These are screening criteria, not an automatic promotion decision.

All these seeds and training pools are search-exposed. Report results as
development evidence. Improvement over FedAvg alone is insufficient to establish
novelty or superiority to FedVal. CIFAR-10, independent validator corruption,
clean drift, final five-seed confirmation, recovery and final EVM agreement are
outstanding. A negative outcome is an acceptable finding; no endless automatic
search or long confirmatory launch follows.

## Integrity and monitoring

Freeze copied package sources, imported coordinator sources, this policy,
configurations, data/provenance hashes, diagnostic artifact and reused artifact
hashes. Preserve original role-based honest-exclusion values, but add a separate
benign-update exclusion metric counting every submitted update in clean runs.
Potential-attacker role labels are not actual attacks in a clean run.

An OS lock prevents duplicate coordinators. Completed cells require configuration
hash validation; incomplete folders are preserved and only that cell reruns.
Outputs: `results/bias-v9-pilot/`. Status, frozen manifest, parity evidence,
analysis and comparison: `artifacts/bias-v9-pilot/`. Launch
`.venv-trust/Scripts/python.exe -m scripts.trust_bias_v9.study` from repo root
in a hidden background process with redirected logs.
