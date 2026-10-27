**ACM TRUST 2027: pilot assessment and proposed full-article study**

Prepared 22 October 2026, Asia/Taipei. This is a review and proposed research
design, not a frozen experiment protocol. No training, registration, submission,
or changes to historical experiments were performed for this assessment.

The strongest current direction is an empirical study of how **private admission
and continued federated training change the effectiveness of backdoor repair**.
V20 is a useful checkpoint-repair module. V21–V23 show why its checkpoint result
does not establish sustained protection. A full article is plausible if controlled,
independent experiments establish this interaction beyond our own method.
The present pilots alone are not submission-ready evidence for a general defense.

Suggested title: **From Repair to Deployment: Private Validation and Persistent
Backdoor Risk in Federated Learning**.

The research question is: *When does a repair that looks effective in isolation
remain useful after private admission and subsequent training?* The proposed
contribution is a reproducible characterization of admission selection and the
tradeoff among repair frequency, clean learning, and reinfection. An improved
algorithm is optional for this evaluation paper; a specific, generalizable
finding is essential. A collection of unsuccessful variants is insufficient.

The best venue fit is **Symposium 5: Test and Evaluation, Benchmarking & Assurance**.
Its [official CFP](https://eigtrust.acm.org/wp-content/uploads/2026/07/CFP_Symposium-5.pdf)
explicitly includes lifecycle assessment, continuous monitoring and reproducibility,
and confirms [CMT Track 10](https://cmt3.research.microsoft.com/TRUST2027/Track/10/Submission/Create).
The [main CFP](https://eigtrust.acm.org/trust2027/cfp/) specifies abstract registration
on **24 October 2026**, paper submission on **31 October 2026**, double-blind
two-column ACM sigconf, and nine pages including appendices, with additional
pages for references and required GenAI disclosure. The cutoff timezone remains
unverified. Symposium 2 is a possible fit if a validated defense becomes the
central contribution. These are full conference research papers.

**What the pilots actually establish**

All latest repair results below use exposed FEMNIST development seeds 11/17/23.
Percentages reported as attack rates use all non-target-label examples; do not
silently replace them with conditional ASR on correctly classified clean inputs.

| Evidence | Measured result | Use in the article |
|---|---|---|
| [V10](MECHANISM_V10_OUTCOMES.md), 42 clean runs | Redistributing validation improves dynamic-trust FEMNIST accuracy by 4.00 pp; fixed trust improves private-validation accuracy by 2.43 pp. Effects differ on Fashion-MNIST. | Motivation for validation-support interventions; not a general claim that trust is harmful. |
| [V13](SECURITY_V13_OUTCOMES.md), 72 runs | Quorum helps against gradual poisoning, but FEMNIST targeted mean/endpoint rates remain 93.21%/99.65%. | Establishes that clean utility and gradual-attack success do not imply backdoor protection. Historical architecture/horizon differs from latest runs. |
| [V19-fixed](../../artifacts/interleave-v19-fixed/comparison.md), 24 checkpoint cells | Sequential to interleaved hard-patch repair changes raw attacked trigger rate 47.45% to 50.50%, while committed rate improves 84.01% to 69.21%. | A concrete repair-ranking reversal after admission; preserve the failed status. |
| [V20](ANCHOR_V20_OUTCOMES.md), 24 checkpoint cells | Full repair reaches 9.23% mean committed trigger rate; all three attacked repairs pass. Clean checkpoint accuracy loss averages 0.95 pp. | Positive checkpoint control, with known in-bank trigger and honest repair assumptions. |
| [V21-fixed](../../artifacts/online-v21-fixed/comparison.md), 12 clean trajectories | Frequent V20 repair loses 2.77–4.20 pp final clean accuracy; 24 attack cells held. | Evidence that a successful endpoint repair does not establish online clean feasibility. |
| [V22](../../artifacts/periodic-v22/comparison.md), 12 clean trajectories | Periodic-only repair loses 0.25–0.31 pp; primary peak gate admits zero of 12 repairs. Its 24 attack cells remain held. | Strict admission can reproduce the unrepaired model by doing nothing. It is not a demonstrated security benefit. |
| [V23](../../artifacts/periodic-attack-v23/comparison.md), six new trajectories and 21 reused rows | Periodic repair ends at 10.07% mean trigger rate, but mean across attacked rounds is 94.45%. All seeds fail the declared security target. | Strongest motivation: favorable endpoint, little protection across the trajectory. |

V23 aggregate means recomputed from its preserved JSON:

| Method | Clean accuracy | Gradual-attack accuracy | Targeted clean-input accuracy | Mean targeted rate, rounds 11–100 | Targeted rate at round 100 |
|---|---:|---:|---:|---:|---:|
| Clipping | 78.65% | 45.70% | 77.00% | 98.61% | 99.80% |
| Current quorum | 78.54% | 67.61% | 77.46% | 97.49% | 99.96% |
| Periodic V20 | 78.25% | 68.79% | 78.30% | 94.45% | 10.07% |

The periodic candidate improves the endpoint by **89.89 pp**, but the temporal
mean by only **3.05 pp**, relative to quorum. Round 100 is itself a scheduled
repair, so the endpoint is measured at a favorable phase. Seed 17 goes from
6.68% after repair at round 75 to 94.23% at round 76. Seed 23 goes from 6.10%
at round 50 to 89.59% at round 51 and 99.69% at round 52. Across seeds, attack
rate is at least 90% in **81, 83 and 80 of 90** attacked rounds.

These are observations under continued poisoning. To separate reinfection caused
by fresh poisoned updates from backdoor persistence or ordinary training drift,
we need matched attacker-stop continuations. Also state the deployment model:
the temporal mean measures exposure if each committed round serves predictions.
For a model deployed only once after training stops, endpoint repair is a different,
potentially useful objective. The current mean assumes equal weight per round;
it is not a measured real-world incident rate.

![V23 committed trajectories and endpoint versus temporal mean](../../artifacts/pilot-review-20261022/v23-temporal-gap.png)

[Recomputed evidence and input hashes](../../artifacts/pilot-review-20261022/evidence.json)
and [reproduction script](../../artifacts/pilot-review-20261022/summarize.py) accompany this review.
The script verifies that all three full histories reproduce their stored means
and endpoints. A separate read-only review checked 36 sealed files across the
six V23 runs and found all hashes matching. Hash agreement establishes artifact
consistency, not independent replication or algorithmic correctness.

One tempting explanation is already contradicted by the evidence: V21 accepted
repairs generally improve immediate clean accuracy, even though final accuracy
is worse. The [trajectory accounting](../../artifacts/online-v21-followup/trajectory-accounting.md)
therefore does not support a story of repeatedly accepting small negative steps.
Intervening on continued trajectories is needed to identify the cause.

**What must be new**

This focused literature check identifies substantial overlap; it does not
establish priority or exhaust all related work.

| Primary source | Existing overlap | Required distinction |
|---|---|---|
| [BaFFLe](https://arxiv.org/abs/2011.02167) | Private client feedback for model admission, including FEMNIST. | Demonstrate how admission selects among repaired candidates under controlled support and skew. |
| [CrowdGuard](https://arxiv.org/abs/2210.07714) | Client feedback and behavior-based filtering. | Compare the object being checked, trust assumptions and actual admission outcomes. |
| [NAD](https://arxiv.org/abs/2101.05930), [ADFL](https://doi.org/10.1016/j.cose.2023.103366) | Teacher-guided backdoor repair; ADFL applies distillation to FL. | V20's frozen parent/output anchoring is an operational difference requiring evidence, not a first-distillation claim. |
| [FLIP](https://arxiv.org/abs/2210.12873) | Client hardening, single-shot and continuous attacks. | Continuous poisoning alone is not a novel evaluation setting. |
| [Neurotoxin](https://proceedings.mlr.press/v162/zhang22w.html) | Backdoor durability after attackers stop. | Separate stopped-attacker persistence from continued-attacker reinfection. |
| [Security-evaluation pitfalls](https://people.cs.umass.edu/~amir/papers/DLSP23-defenses.pdf) | Evaluation choices can change FL robustness conclusions. | Establish a specific repair/admission/frequency interaction beyond a generic call for better evaluation. |

The three-stage accounting is useful: for admission indicator a, candidate attack
rate R and parent rate P, committed rate is aR + (1-a)P on the same examples.
This is an identity, not a new theorem. The contribution would be evidence that
private validation changes repair rankings systematically, and that those
changes interact with continued training under controlled budgets.

**Proposed experiments, in decision order**

First freeze the deployment definition, exposure ledger, evaluation splits,
outcomes and baseline implementations. Keep all historical pilots as development.
Fresh seeds alone cannot undo repeated tuning on the same evaluation examples.
Audit unused writers/data and use untouched evaluation partitions; where none
exist, obtain new held-out data and disclose the earlier reuse. Preserve fixed
client evaluation groups across validation interventions.

1. **Isolate admission on fixed candidates.** On natural-client FEMNIST and a
   controlled-skew Fashion-MNIST setup, generate candidates with clean tuning,
   V20 and a faithful established repair such as NAD. For each identical parent
   and candidate, vary validator support/skew and sample count without retraining
   the candidate. Compare private panels with a more representative allocation
   of the same validation-only pool. Fix panel counts and total unique examples
   within each support comparison. Repeated panel draws are nested measurements,
   not independent training seeds. Record private/global clean-utility disagreement,
   rejected security gains, and raw/committed method rankings. A global-clean-data
   gate is an offline diagnostic only; it must not use the test set or target labels
   to select the deployed model. Calibrate operating points on development data,
   then freeze them; do not retrospectively equalize acceptance on the test results.

2. **Separate repair from reinfection.** From the same poisoned parent and complete
   saved state, branch into five policies: no repair; V20 with current admission;
   V20 forced admission; NAD with the same admission wrapper; NAD forced admission.
   Forced admission and the NAD wrapper are explicitly diagnostic adaptations.
   Cross these with attacker stops versus attacker continues. With two datasets
   and three development seeds this is **60 continuation branches**, plus any
   source-checkpoint generation. Evaluate the next 1, 2, 5, 10 and 25 rounds.
   Keep future client sampling and ordinary-training budgets matched; gradients
   naturally diverge after model intervention. Include optimizer/RNG state in
   checkpoint identity. Reject-all/no-repair is essential to expose all-hold gates.

3. **Separate repair frequency from extra optimization.** Compare 4, 8 and 16
   interventions with both 200 updates per attempt and 800 total repair updates.
   The four-intervention setting is shared, giving five unique configurations.
   Two datasets, clean/continuous-attack conditions, three development seeds give
   **60 trajectories**. Freeze exact schedules, common warm-up and a common first
   repair before execution; schedules must not be selected from outcomes. Match
   teacher, data access and all other loss settings. Count actual proposal,
   teacher and verifier work, because equal optimizer updates do not equal FLOPs.
   Evaluate every committed round and a fixed post-final-repair continuation;
   do not compare endpoints at arbitrarily different schedule phases. This stage
   estimates a frequency/utility/security tradeoff, not an optimal schedule theorem.

4. **Run a focused independent core only after the mechanism and baselines are
   credible.** A proposed matrix is below. It is a research-design recommendation,
   not a venue-required number of runs or an already authorized launch.

| Dimension | Proposed frozen core |
|---|---|
| Datasets | Natural writer FEMNIST; Fashion-MNIST with controlled non-IID partitions. Add CIFAR-10 as the first transfer extension if making broad vision claims. |
| Methods | Clipping; current quorum; quorum plus compute-matched clean tuning; periodic V20; faithful NAD repair adapter; FLIP; BaFFLe. |
| Conditions | Clean; poisoning that stops at a prespecified round; continuous targeted poisoning; gradual untargeted poisoning. |
| Replication | Five fresh, exposure-audited paired seeds as an initial precision floor, with independent evaluation data. |
| Size | 2 datasets × 7 methods × 4 conditions × 5 seeds = **280 trajectories**. FedAvg sanity controls add 40 if included. |
| Training | Start from the existing 50-client/40%-participation setup; choose horizon/model settings after development convergence checks and freeze before confirmation. |

Validate native baseline behavior and clean convergence before comparing attack
outcomes. NAD is not originally a federated admission algorithm: document its
teacher training, reference-data access and wrapper, and preserve a native
checkpoint comparison. Run FLIP/BaFFLe under their documented assumptions; report
native and resource-matched sensitivities separately. Do not cripple their
reference data or tune them less carefully to manufacture an advantage.
The existing FedVal adapter is a useful additional sensitivity after fidelity
and clean-convergence checks, especially if heterogeneous aggregation becomes
central. FedRecover/Crab are essential empirical comparisons only if the article
claims identified-attacker rollback/recovery superiority.

5. **Test the boundary on frozen stress cases.** For the four relevant methods
   (quorum, periodic V20, FLIP, BaFFLe), add two conditions: an effective trigger
   family outside the repair bank, and a defense-aware attacker optimized on
   separate proxy data against the known gate/schedule. Two datasets × four methods
   × two conditions × five seeds is **80 additional trajectories**, with matched
   effective-attack controls. Freeze attacker budget, knowledge, target and search
   procedure before evaluation. Do not select only the strongest result after
   test access. Include corrupt repair providers/validators only if expanding the
   threat model; otherwise retain the honest-provider limitation prominently.

The expensive confirmation should not precede a clear novelty decision. If the
interaction disappears with faithful external repairs or representative data,
report that result and narrow the claim; do not keep changing the candidate until
it passes. Historical held V21/V22 attacks stay held. Any diagnostic evaluation
of a clean-harmful arm belongs to a separately declared failure study.

**Outcomes and analysis to freeze**

The primary security outcome is the mean committed attack rate over a fixed
post-onset deployment horizon. Also report raw candidate rate, post-admission
rate, endpoint rate, fraction of rounds above a declared unsafe threshold,
rebound after each admitted repair, and a complete attack-stopped continuation.
The 20% threshold used in these pilots is an experimental operating point,
not a universal safety criterion. Include conditional ASR with its denominator
as a separate metric. An ineffective attack is inconclusive evidence of defense.

Utility outcomes are final and trajectory clean accuracy, fixed-client macro and
lower-decile accuracy, honest-update exclusion and admissions/holds. Cost includes
repair/teacher/validator examples, unique data, local training, bytes and isolated
runtime. Modeled bytes do not establish network latency. Keep the passive V23
monitoring cost or separately verify a version that removes it.

Use per-seed paired effects and uncertainty; training rounds, examples and panel
votes are correlated, not extra independent runs. Five seeds may be too few for
small effects; a two-sided exact sign-flip test with five pairs cannot yield
p < .05. Plan precision or expand the independently sampled seed set before
confirmation if inferential claims require it. Freeze primary contrasts and
handle multiple secondary comparisons transparently. Preserve every completed,
failed and held cell in the completeness report.

**Reuse, exclusions and manuscript scope**

Reuse the tested harness, frozen data provenance, exact commit checks, raw versus
committed evaluation, V20 module and matched V16-fixed controls. Preserve historical
implementation identities. Initial V15/V16 integrity failures, original V19 parity
failure and original V21 missing-field failure are software/provenance exclusions,
not scientific defense outcomes. Their fixed successors must remain separately
labeled. Earlier V9 accounting/evaluation defects were addressed in the later
inherited code; do not report those old defects as current V23 bugs.

Latest runs use memory-ledger, validation-only, fixed-trust settings and zero
malicious validators. Correct-label honest repair and truthful private panels
are assumed even for clients malicious during ordinary training. Roles/examples
can recur across rounds. These results support neither blockchain-derived security,
Byzantine repair tolerance, economic deterrence nor formal privacy. Keep blockchain
as optional provenance infrastructure unless a distinct enforcement experiment
demonstrates why it is necessary. SIFT/EMH routing and stake attribution would
distract from this nine-page article's central question.

The existing manuscript still centers two-layer accountable recovery. It needs
a substantive rewrite after the claim decision. A nine-page allocation is:
problem and closest work 1.5; setting and threat model 1; methods and measurement
protocol 1.5; controlled mechanism results 2; independent/stress results and cost
2; limitations and conclusion 1. Four useful figures are admission-induced ranking
changes, temporal attack trajectories, the repair-budget tradeoff, and independent
transfer/stress results. Keep the detailed exploratory history in the artifact.

**Decision for the remaining submission window**

On October 22–23, settle the novelty claim, audit baseline feasibility and untouched
data, and perform the bounded mechanism work if pursuing this direction. Benchmark
the final baseline implementations before forecasting the large matrix: the six
V23 runs taking about 15 minutes does not estimate FLIP/NAD/CIFAR costs. Before
the October 24 abstract cutoff, describe only completed findings and the intended
question; registration remains a separate author action. October 24–28 is the
possible confirmation window, with October 29–30 reserved for analysis, writing
and checks. There is no evidence yet that all necessary implementation and
confirmation work will fit this window.

Proceed to a full article only if the controlled effect survives independent
data, applicable external comparisons and the declared measurement protocol.
If only the existing three-seed in-bank result remains, it is a useful pilot
finding but insufficient support for the proposed full research contribution.
