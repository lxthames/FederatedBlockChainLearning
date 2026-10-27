# Literature pilot implementation and fidelity audit

19 October 2026. Exploratory implementations in `scripts/trust_literature_v6/`.
All earlier packages/runs remain frozen. One development seed cannot validate a
final protocol or provide seed-level confidence intervals. This pilot selects
research directions, not manuscript claims. No baseline is silently substituted.

## Source mapping

- FLAME: [USENIX Security 2022 paper](https://www.usenix.org/system/files/sec22-nguyen.pdf),
  Algorithm 1, section 4.3 and appendix B. `literature.flame` computes cosine
  distances between full local models; HDBSCAN uses strict-majority minimum
  cluster size, min_samples=1 and allow_single_cluster. It clips admitted deltas
  at the median norm of all deltas, averages uniformly and adds Gaussian noise
  with sigma=.001 times the bound. The image-scenario noise multiplier is a
  development setting, not a privacy budget for this dataset. No formal DP or
  universal backdoor-removal claim follows. All-noise/no-majority clustering is
  an explicit failed run, never FedAvg fallback. Tests check its components.
- FLShield: [IEEE S&P 2024 author paper](https://arxiv.org/pdf/2308.05832),
  sections 5.1.2--5.4 and appendix 1.5. The paper-based cluster version uses
  KMeans/silhouette, representative means, class-wise loss impact, iterative
  imputation, EllipticEnvelope report filtering, minimum class scores, top-half
  representative selection and norm clipping. Exact Gram coordinates preserve
  update-space distances while reducing KMeans cost. n1=10 (strictly more than
  ten examples), n2=30; unsupported classes raise. All 50 clients validate each
  representative in this pilot; communication/evaluation counts include this
  larger budget. Underspecified choices are explicit: contamination=.1,
  covariance support_fraction=.75, ten KMeans restarts, deterministic streams,
  all cutoff ties retained. Covariance warnings are logged. This is a clean-room
  paper-based implementation, not verified author-code equivalence. Honest
  validators only in this pilot; nonzero validator corruption raises.
- FLDetector: [KDD 2022 paper](https://arxiv.org/pdf/2207.09209) and
  [official MXNet repository](https://github.com/zaixizhang/FLDetector).
  `PartialFLDetector` is an adaptation named `fldetector_pp`: per-client secants
  across participation gaps, negative local-SGD deltas as gradient proxies,
  matrix-free Hessian prediction, normalized residuals, five-observation
  histories, gap statistics with k=1,2 and ten reference draws. At least three
  residual observations and three successive high-suspicion observations are
  required before persistent quarantine. Positive-curvature safeguards count
  rejected pairs. Admitted updates use coordinate median. This is not the
  original full-participation algorithm, which uses its own history construction
  and restart behavior. Pilot records honest update exclusion and suspects.
- FedRecover: [IEEE S&P 2023 author paper](https://arxiv.org/pdf/2210.10936),
  algorithms 1--3. `fedrecover_partial` is an adaptation: local-SGD deltas and
  recorded partial participation replace single-gradient/full-client requests.
  It restores genesis, uses six warmup steps, three-secant Hessian products,
  five-step corrections, three final exact steps, and exact fallbacks for absent
  curvature or estimates exceeding twice the archived update's infinity norm.
  Pair buffers use recovered/original model and gradient differences. Curvature
  safeguards and the stated threshold differ from an exact reproduction; all
  exact/estimated requests and rejected pairs are reported. No convex-theory
  guarantee is asserted for CNN local training. Exact retraining is a comparator.
- Crab: [author paper](https://arxiv.org/pdf/2401.08216), algorithms 1--2.
  `crab_fixedwindow` is an adaptation using fixed five-round windows, top .6 KL
  round retention, top .7 cosine contributor retention, ratio .3 for cumulative
  mean-update sensitivity, and norm-calibrated local retraining. KL uses disjoint
  server-reference feature probes. The paper's loss-adaptive windows and
  influence definition are not reproduced. Selected original checkpoints may
  remain poisoned. Shared full history stays on disk; selected-file bytes are a
  logical upper bound, not demonstrated physical storage savings or certification.

## Data, scheduling and budgets

Only the existing deduplicated official Fashion-MNIST training-pool development
arrays for seed 11 are loaded. Training/private validation/reference partitions
use the standard disjoint splitter on this pool; held-out development evaluation
is supplied separately. Official test arrays are not loaded. Previous development
exposure and search remain disclosed; seed 11 is not confirmatory.

18 online runs: six methods x clean/gradual/targeted replacement. Methods are
FedAvg, rank/trust without recovery, V5 paired-loss recovery, FLAME, paper-based
FLShield clustering, and adapted FLDetector. Defaults: 50 clients, .4 participation,
.3 attackers, Dirichlet alpha .5, 30 rounds, one epoch, batch 64, SGD .01 with
momentum .9, attack onset 11. Same partitions/client selection/training/attack
streams across defenses; distinct clustering/filtering/noise streams. Monetary
recovery penalties are disabled. This delays attacks and does not replace V5's
immediate-attack evidence. These new pilot settings are not pooled with V5.

All clean cells run first. The FedAvg clean utility must reach .20; candidates
must lose <=.05 utility, have <=3 unsupported rounds and zero honest penalties.
Otherwise attacked cells are held. This gate is an inexpensive development
screen. Attack effect is separately checked against FedAvg and clean trigger
predictions. Failed/held methods remain visible; a completed pilot can contain gaps.

The pilot waits for V5 to finish before training, keeping at most the original
classification study plus this new pilot on the GPU. OS coordinator lock prevents
duplicate writers. Code, audit, manifest and data are frozen before launch.
Interrupted incomplete directories are preserved with a timestamp before restart.
Completed online cells require matching configuration hashes for reuse.

## Conditional recovery experiments

All FedAvg runs save off-chain pre-round model states and received updates, plus
the final model. These archives carry no simulator attack labels. Archived
FedAvg updates are passed through the adapted detector after training. Its
suspects are shared across retraining, FedRecover-adapted and Crab-adapted
branches for each attack: six possible detector-selected branches. Missing
suspects yield explicit unsupported results, not an oracle substitution.

Three separately labeled oracle diagnostic branches use the targeted attack's
known identities solely as conditional recovery upper bounds. They cannot
validate detection, select an end-to-end defense, or justify penalties. Oracle
information is part of their input and is excluded from the detector branch.
All nine possible branches share their source history and identity set within a
comparison. They run after the source training horizon, not inside the committee
protocol. The original source ledger/trace stays append-only; branches have
separate recovery traces and source hashes. No new EVM deployment is claimed.

Remaining malicious clients continue their attack behavior during exact recovery
requests. The simulator callback alone uses attack identities; model selection
and aggregation do not inspect them. Some omniscient gradual attack requests may
be unsupported if an exact-request subset has no honest reference; report that
failure. Extra training, communication estimates, runtime, archive bytes, false
quarantines and residual ASR are reported. On-chain receipt costs are not mixed
with these off-chain estimates.

## Verification and decision gate

Tests check Hessian products against explicit dense BFGS, clipping/noise, missing
class support, report filtering, participation-gap histories, future-response
causality, private split integrity, held-out-label independence, append-only
source preservation, oracle/detector separation and recovery work accounting.
Two-round full-development-data benchmarks precede launch for a runtime forecast;
their outputs are excluded from the online grid. They do not tune parameters.

`artifacts/literature-v6-pilot/comparison.md` and `analysis.json` compare matched
online cells. At least .01 final utility gain for gradual attacks or .05
attacked-round ASR reduction for targeted attacks, clean-screen success and
attacked utility loss <=.05 merit replication review only. One seed, covariance
warnings, adaptation differences, untested validator corruption/adaptive evasion
and incomplete cells remain limitations. The report never automatically promotes
a method or launches confirmation. Recovery diagnostics are reported separately.

Initial two-round profiling produced IterativeImputer convergence warnings.
The final runner records these and covariance warnings in structured events and
summary counts. They require numerical review before selecting a publication
baseline. The imputer's fixed iteration budget is not silently increased based
on evaluation utility. Runtime benchmarks use source-hash-specific directories;
older profiling artifacts remain preserved and excluded from the pilot grid.
