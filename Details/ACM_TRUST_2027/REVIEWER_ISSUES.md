# Reviewer issue tracker

## Evidence update: 19 October 2026

Literature follow-up: a bounded V6 Fashion-MNIST development pilot now implements
paper-based FLAME/FLShield comparators, an explicitly adapted partial-client
FLDetector, and separate conditional retraining/FedRecover-/Crab-adapted branches.
See the [source-to-code audit and frozen pilot design](../../docs/research/LITERATURE_V6_BASELINE_AUDIT.md).
No new method is promoted into publication comparisons before the pilot and
fidelity review. Oracle recovery rows are diagnostics only. Structured imputation
and covariance warnings, failed cells, missing suspects and extra recovery costs
must accompany any resulting comparison. This pilot does not resolve independent
validator-corruption, cross-dataset or complete-seed evidence requirements.

Recovery effectiveness remains unresolved. The completed V4 development study
had 27 successful executions and no recovery events; neither recovery variant
improved the rank/trust control. This preserves the negative result, rather than
treating a completed execution as successful defense validation. See
[V4 design and limits](../../docs/research/GUARDED_RECOVERY_V4.md) and
`artifacts/recovery-v4-diagnosis/report.json` for the event-level diagnosis.

A bounded V5 diagnostic compares recovery disabled, the previous accuracy-plus-
benefit rule, and direct paired-loss quorum under immediate and delayed attacks.
It uses one development seed and cannot settle the reviewers' robustness concerns.
[Design, mathematical rule and assessment policy](../../docs/research/LOSS_MONITOR_V5_PILOT.md)
specify its limits. The broader extension confirmation remains paused. Final
manuscript equations, results and protocol version must be reconciled after a
defensible final design is selected; original-version results cannot validate V5.

The dedicated SIFT1M study has finished with 211 successful runs and five failed
runs (33 complete five-seed case-study groups). Failed groups must remain visible
and cannot be reported as complete comparisons. Original classification execution
continues separately; counts below describe earlier work and are not live status.

Implementation completion and experimental resolution are different statuses.
The earlier small-client pilot and background synthetic diagnostics cannot close
the publication-evidence requirements below.

| Review issue | Implemented revision | Required evidence / current status | Manuscript section |
|---|---|---|---|
| R1: only 4/8 clients, 20 rounds | TRUST matrix: 50-client core, 20/50/80 sensitivity, 100 rounds | Confirmatory runs pending | Experimental Protocol; Results |
| R1: ambiguous 30% committee and 2/3 voting | Proposer exclusion; min-three committee; ceiling quorum; unsupported rounds | Unit and deployed-contract checks pass | Layer 1 |
| R1: datasets, attacks, Multi-Krum, FLTrust | Two official classification datasets; six matched methods; five conditions | Development validation and 300 core runs pending | Experimental Protocol |
| R1: rollback conflicts with immutability | Restore off-chain checkpoint; append recovery and penalty events; retain prior versions | Contract tests and live training integration verified | Layer 2; Implementation |
| R1: adjacent-round false alarms and gradual attacks | Fixed panel, five-observation trend detector, adjacent ablation; bounded reversed-update ramp | Benign drift, FPR/FNR and attack validation pending | Monitoring; Evaluation |
| R2: committee, on-chain aggregation and head scaling | Explicit computation boundaries, gas receipts, communication accounting, 129N head formula | Complete five-round development traces replayed at 20/50/80 clients; real-data scaling pending | Implementation and Complexity |
| R2: 100% initial ASR and overstated final robustness | Distinct acceptance and targeted ASR; pre/post recovery utility; withdraw zero-loss assertion | Matched attack outcomes pending | Results; Limitations |
| R2: unclear Sybil enrollment | External registrar; duplicate attempts and unique colluders modeled separately | Enrollment error tests pass; sensitivity pending | Threat Model |
| R2: heterogeneous Table III metrics | Remove cross-paper rankings; generate matched comparisons only | Five-seed groups required; pending | Results |
| R2: grammar and typos | New prose, consistent terminology and equations | Draft proofread; final author review pending | Entire paper |
| R3: limited novelty | State contribution as executable validation and accountable recovery under explicit assumptions | No fundamental-algorithm or priority claim | Introduction; Related Work |
| R3: inadequate experiments | Shared core/sensitivity matrices and immutable artifacts | Pending publication evidence | Evaluation; Results |
| R3: unjustified trust threshold | Configurable trust floor, eligibility, rewards and penalties; threshold sensitivity | Development and sensitivity results pending | Layer 1; Evaluation |
| R3: overlapping Figure 1 labels | Redraw workflow using separate sized blocks | Render-and-inspect required at each material revision | Figure 1 |
| R4: non-IID local validation bias | Disjoint private validation; fixed panels; separate held-out worst-client utility; disclose bias | Heterogeneity/drift outcomes pending | Threat Model; Monitoring; Limitations |
| R4: SIFT-only and small scale | Classification core; new SIFT1M supporting study with exact neighbor-derived labels | New labels verified; matched experiments running; original labels excluded | Evaluation |
| R4: EMH contribution and deployment barriers | Actual two-branch model option; 128-dimensional input; MLP comparison; registrar/orchestrator assumptions | New SIFT1M case study running with separate development and evaluation seeds | Implementation; Limitations |
| R4: recent literature missing | BOBA 2024, Crab 2024, VerifBFL 2025, FAUN 2026; verify primary metadata | Bibliography reviewed against primary pages; final author review pending | Related Work |
| R4: unclear figures and presentation | Editable TikZ figures, separate sections, ACM bibliography | Final visual and wording checks pending | Entire paper |

## Prospective TRUST review risks (19 October 2026)

The original reviews are not exhaustive. See the
[strengthening strategy](../../docs/research/TRUST_2027_STRENGTHENING_STRATEGY.md)
and [primary-source literature assessment](../../docs/research/TRUST_2027_LITERATURE_ASSESSMENT.md)
for prioritized additions. Implementation has progressed, while evidence remains pending:

- Paper-based FedVal and matched recovery-only controls are implemented in a
  versioned follow-up study; their completed results remain pending.
- Bounded adaptive evasion, false recovery attestations, vote withholding,
  honest-client exclusion metrics and EVM trace agreement checks are implemented.
  Their scientific interpretation still requires completed evidence.
- The 62-class writer-based FEMNIST data is prepared for all eight seeds,
  checked against the official archive checksum and deduplicated across pixel
  content before training/private-validation partitions are formed. The new
  study is running: 33 development cells and up to 275 evaluation cells.
- The follow-up adds an explicit local-attestation recovery quorum guard.
  Original core results cannot validate that changed protocol; final-version
  coverage and manuscript claims must be reconciled before submission.
- Audit the fixed-z, honest-reference ALIE variant and its failed development
  cells before reporting it as a faithful standard attack reproduction.

See the [extension runbook](../../docs/research/TRUST_EXTENSION_RUNBOOK.md)
and [baseline/protocol audit](../../docs/research/TRUST_EXTENSION_BASELINE_AUDIT.md).

## Final evidence gate

### Evidence update: 19 October 2026, after V6 completion

The earlier running-study notes above are historical. Current outcomes and
limitations are recorded in the [V6 outcomes memo](../../docs/research/LITERATURE_V6_OUTCOMES.md).

| Concern | Implemented work and observed evidence | Remaining manuscript/evaluation obligation |
|---|---|---|
| Validation bias and honest exclusion | Rank/trust Fashion-MNIST seed 11: clean 75.9%, gradual 68.7%; honest exclusions retained in logs | V7 development replication on seeds 17/23; matched comparisons and heterogeneity limits in Evaluation/Results |
| Detection/recovery effectiveness | V5 completed 15 cells with zero recovery; V6 recovery supplied no additional online benefit; targeted archive detector found 2/15 attackers | Explicit negative/conditional findings in Recovery/Results/Limitations; no claim of reliable poisoning attribution |
| Literature fidelity and novelty | FLShield/FLAME and partial FLDetector/FedRecover/Crab implemented with fidelity audit; six attack cells held after clean-gate failures | Disclose adaptations, warnings, validation costs, and overlap in Related Work/Implementation; median control for detector |
| Economic accountability | V7 keeps monetary poisoning penalties disabled; oracle identities labeled diagnostic only | Honest approval is not proof of culpability; no economic-deterrence claim |
| SIFT evidence completeness | Study finished with 211 completed cells and five failed cells, rather than still running | Use complete five-seed groups only; disclose failed EMH/FLTrust gradual cells |
| Traceable complete classification evidence | Original workflow stopped after 379 completed/eight failed cells, with artifact-generation error | Diagnose interruption before resume; reconcile original-v2 evidence with selected final protocol |

The [V7 bounded replication protocol](../../docs/research/RANK_REPLICATION_V7.md)
freezes 13 new GPU runs and seven reused rows. It does not constitute a final
protocol freeze or submission readiness.

V7 has now completed all 13 new runs without failed/held cells. The
[V7 outcomes and diagnostic correction](../../docs/research/RANK_REPLICATION_V7_OUTCOMES.md)
report three-seed mean gradual utility 69.73% versus FedAvg 26.61%, and targeted
mean ASR 14.11% versus 50.76%. Clean utility is 74.14% versus 74.63%; honest
update exclusion remains 17.6–22.4%. The median control explains most of the
detector's clean utility gap, but a clean client was falsely quarantined.
Recovery benefit and final protocol confirmation remain unestablished.

The next finite batch is specified in the
[V8 comparison and candidate review](../../docs/research/COMPARISON_V8.md):
26 new Fashion-MNIST median/FLTrust/paper-based FedVal cells and up to 18
natural-client FEMNIST FedAvg/rank cells, reusing 19 completed rows. The
candidate algorithm is unchanged. This addresses matched baseline strength
and natural heterogeneity; it leaves CIFAR-10, final-version confirmation,
validator corruption, recovery efficacy and deployment reconciliation open.
No claim should treat validation-only rank/trust as validated full recovery.

V8 is complete with 40 successful cells, zero execution failures and four held
FEMNIST rank attack cells. The
[V8 outcomes](../../docs/research/COMPARISON_V8_OUTCOMES.md) show that paper-based
FedVal outperformed rank/trust on the measured Fashion-MNIST endpoints, while
rank failed FEMNIST clean gates on seeds 11/17. These findings block promotion
of the current candidate. The
[V9 validation-bias pilot](../../docs/research/BIAS_V9_PILOT.md) tests smaller
probes, observed-class balancing and uncertainty backoff against a clipping
control, with mandatory unchanged-rule parity. Its results are pending. The
denominator correction for benign-update exclusion is documented explicitly.

Keep this draft unpublished until the evidence-status report, failed/held
experiment logs, final manuscript claims, anonymity, page budget and rendered
pages have been reviewed. Contract gas results are local EVM measurements;
they do not validate network finality or economic deterrence.
