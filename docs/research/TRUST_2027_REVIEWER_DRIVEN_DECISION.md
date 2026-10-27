# TRUST 2027: reviewer-driven decision after V9

Assessment date: 19 October 2026 (Asia/Taipei). V9 finished at 21:32 Taipei.
This is a strategy and evidence assessment, not a new frozen experiment manifest.
No training, coordinator, contract, historical policy or result was changed.

## Recommendation

Do not expand confirmation of the current rank/trust candidate. V8 and V9 expose
a clean-utility failure on natural-client FEMNIST; FedVal is stronger on the
measured Fashion-MNIST endpoints; recovery has no demonstrated incremental
online benefit. More datasets or additional runs of the same candidate would
not resolve these problems.

The most promising research question is: **when does heterogeneous private
validation confuse legitimate distribution differences with harmful updates,
and does reputation feedback amplify that mistake into exclusion and loss of
committee availability?** Establish the mechanism with controlled interventions,
then evaluate one bounded mitigation only if those interventions justify it.
An alternative paper could be a rigorous failure-boundary/evaluation study if
the mechanism generalizes and its novelty is established. A collection of failed
experiments is not sufficient by itself.

## Venue and timing

The repository targets ACM Conference on Trustworthy and Responsible AI and
Computing Systems (ACM TRUST 2027). The official
[CFP](https://eigtrust.acm.org/trust2027/cfp/) currently lists abstract registration
on October 24, 2026, paper submission on October 31, and the conference on
March 7-9, 2027. The exact submission cutoff timezone was not established here.
Secure and Resilient AI Systems is a good thematic fit; Evaluation, Benchmarking
and Assurance also fits a mechanism/evaluation contribution. These are scope
recommendations, not confirmation of an existing CMT track selection.

The [author guidelines](https://eigtrust.acm.org/trust2027/authors-guidelines/)
require completed, validated work, double-blind ACM sigconf format, and at most
nine pages including appendices. References and GenAI disclosure may extend
beyond that limit. Review criteria include originality, correctness, significance,
clarity and reproducibility. Acceptance cannot be estimated from the available
evidence.

## Verified progress and its limits

Live status files take precedence over older README statements that jobs are
still running. No Python processes appeared in the process inspection during
this assessment.

| Study | Recorded execution | Scientific meaning |
|---|---|---|
| Original classification workflow | 379 completed, 8 failed; stopped in artifact generation | Substantial older-protocol evidence, not final-protocol completion |
| Original manuscript core report | 294/300 runs, 58 complete five-seed groups | Six missing ALIE runs; report belongs to the earlier implementation |
| SIFT1M | 211 completed, 5 failed; 33 complete five-seed groups | Supporting routing study; failed groups must remain visible |
| Guarded recovery V4 | 27 completed | No recovery events; no demonstrated benefit over its control |
| Loss monitoring V5 | 15 completed | No recovery events; completion does not validate recovery |
| V7 replication | 13 new completed runs | Promising three-seed Fashion-MNIST development results |
| V8 comparison | 40 completed, 4 held, 0 execution failures | Stronger FedVal comparator; FEMNIST exposes clean harm |
| V9 bias pilot | 24 completed, 18 held, 0 execution failures; both parity checks passed | All three validation variants fail the clean gate; clipping alone does not stop targeted attacks |
| Earlier GPU extension | Stopped for readiness audit after 33 development cells | Do not treat its planned 275 evaluation cells as completed |

Sources: `artifacts/trust-2027/status.json`,
`Details/ACM_TRUST_2027/evidence-status.json`, `artifacts/sift1m-study/status.json`,
`artifacts/{guarded-recovery-v4,loss-monitor-v5-pilot,rank-replication-v7,comparison-v8,bias-v9-pilot}/status.json`,
and `artifacts/trust-extension-v1-gpu/status.json`.
These counts overlap through reuse and must not be summed as independent evidence.
The old manuscript's false SIFT-provenance flag checks an older data root; the
dedicated verified SIFT1M study is the relevant replacement.

Three-seed **development** means, recomputed from
`artifacts/bias-v9-pilot/analysis.json` (including its explicitly reused rows):

| Fashion-MNIST endpoint | FedAvg | Rank/trust | Paper-based FedVal |
|---|---:|---:|---:|
| Clean utility | 74.63% | 74.14% | 75.46% |
| Gradual-poisoning utility | 26.61% | 69.73% | 69.82% |
| Targeted-poisoning utility | 64.03% | 72.47% | 74.01% |
| Mean post-onset targeted ASR | 50.76% | 14.11% | 5.47% |

The 0.09-point gradual difference is descriptive, not a significant win.
FedVal has different reference-data access and validation cost; these are not
equal-information comparisons. Its adapter is paper-based, not certified
equivalent to the released author implementation. FLTrust's approximately 20.6%
clean utility in V8 requires a convergence/fidelity diagnosis before using it
as evidence of superiority.

| FEMNIST method | Three-seed clean utility | Interpretation |
|---|---:|---|
| FedAvg | 53.69% | Control |
| Original rank/trust | 46.85% | Mean loss 6.84 points; seeds 11 and 17 fail the five-point gate |
| Smaller probe | 40.35% | Failed clean gate |
| Observed-class-balanced probe | 44.00% | Failed clean gate |
| Uncertainty-backoff probe | 46.82% | Failed clean gate |
| Clipping only, fixed trust | 53.67% | Clean gate passes; targeted ASR remains 97.38% |

Clipping's gradual utility is 19.34% versus FedAvg's 14.22%, but its targeted ASR
is 97.38% versus 97.18%. This control preserves clean utility without solving
the targeted attack. The 18 held V9 attacks belong to the three failed variants;
do not infer their attack performance or reinterpret the holds as execution errors.

The diagnosis records FEMNIST low-overlap approval at 55.7-62.8%, versus
94.1-97.5% for high-overlap proposals. Approximately 22.0-23.4% of submitted
benign updates were excluded in the three clean runs. These observations are
consistent with validation bias but do not establish its causal role. Votes
share models, rounds and examples. Clean benign-exclusion denominators include
all submitted clients; older honest-role metrics exclude potential-attacker roles.

## Most consequential Globecom objections

Review provenance: the local `Details/GlobalCom_2026.pdf` is the submitted paper,
not review reports. The connected-mail search returned no Globecom review
messages. The mapping below is from
[the repository reviewer tracker](../../Details/ACM_TRUST_2027/REVIEWER_ISSUES.md),
not independently recovered verbatim reviews. No numeric scores or ordering of
individual reviewers' harshness can be established. Priority is my assessment
of the technical objections in light of current results.

| Priority | Tracked objection | Why it remains consequential | Decisive response |
|---|---|---|---|
| 1 | R4: non-IID local validation bias | FEMNIST now shows the exact failure feared | Separate validation support, filtering and trust feedback; report clean client harm and lower-tail utility |
| 2 | R3: limited novelty; unjustified trust threshold | Known components and a FedAvg win do not establish a contribution | Establish a distinct feedback mechanism and boundary; isolate what trust adds beyond clipping/ranking/FedVal |
| 3 | R2: initial 100% ASR, overstated final robustness and heterogeneous comparisons | Targeted ASR, update acceptance and recovery are different outcomes | Matched trajectories, real attack success, clean controls and honest-exclusion costs; remove zero-loss claims |
| 4 | R1/R3/R4: small scale, short runs and narrow evaluation | New evidence is substantial, but latest candidate studies are 30/40-round development pilots | One frozen final version, 100-round convergence checks, complete independent seed groups, natural-client evaluation |
| 5 | R1/R2: quorum, rollback/immutability, blockchain cost and enrollment | A ledger records decisions but cannot make biased assessments truthful | Exact invariants and trust boundary, same-trace memory/EVM comparison, measured cost and explicit registrar assumptions |

Presentation defects matter, but correcting figures and grammar cannot compensate
for the first three issues. The current draft still centers accountable recovery
and contains pending-result prose; it must be rewritten around the selected
contribution after the evidence decision.

## Novelty boundary and a useful mathematical illustration

Do not claim discovery that heterogeneity harms robust aggregation:
[BOBA](https://proceedings.mlr.press/v238/bao24a.html) explicitly studies selection
bias under label skew, and
[FedVal](https://www.usenix.org/conference/usenixsecurity23/presentation/valadi)
uses server validation to address robustness and group bias.
[Zeno](https://proceedings.mlr.press/v97/xie19b.html) already supplies
loss-based update ranking, while
[Fixing by Mixing](https://proceedings.mlr.press/v206/allouah23a.html) studies
Byzantine robustness under heterogeneity. These primary sources establish
overlap, not a complete novelty search or reproduction audit.

A potentially distinct contribution is the interaction between private
validation disagreement, trust-driven voting eligibility, and persistent
exclusion/liveness. This is a hypothesis to establish, not a verified novelty claim.

A simple counterexample explains why honest votes need not reflect global utility.
Let L1(w)=(w-1)^2/2, L2(w)=(w+3)^2/2, and L=(L1+L2)/2. At w=0 an update
delta=-epsilon gives Delta L1=epsilon+epsilon^2/2 > 0, while
Delta L=-epsilon+epsilon^2/2 < 0 for 0<epsilon<2. A locally harmful update can
therefore be globally helpful. This is a sign-disagreement example, not a proof
that the implemented rank rule necessarily rejects that update or a new theorem
about its convergence. Increasing the quorum does not repair shared objective bias.

## Experiments in decision order

### A. First establish the mechanism: at most 36 clean development runs

Use Fashion-MNIST and natural-client FEMNIST, three development seeds, and a
2x2 intervention:

1. Validation distribution: original private partitions versus a controlled,
   more representative repartition of the **same validation-only examples**.
2. Trust: original dynamic trust versus fixed trust, with identical initial
   weights and zero reward/rejection updates in the fixed arm.

This gives 2 datasets x 2 distributions x 2 trust settings x 3 seeds = 24 runs.
Add FedAvg and clipping-only controls on both datasets/seeds: 12 runs, for 36
logical cells. Reuse existing cells only if full configuration, source, data
and artifact compatibility is demonstrated. Do not silently treat an old variant
as the newly frozen implementation.

Keep training partitions, client identities, examples per validator, total unique
validation examples, scoring calls, update clipping, rank fraction and proposed
participant schedule matched. Redistribute validation examples without replacement;
do not use held-out test examples or claim complete class coverage where the
pool cannot provide it. The representative arm changes data-access assumptions
and is a **diagnostic control**, not a deployable private-data solution.

First replay fixed proposals on common checkpoints to measure the immediate
vote effect. Then retrain end to end to measure accumulated effects. Do not force
identical committee membership after trust diverges: membership is an outcome
of the intervention. Report offered participation separately from accepted
updates, validator eligibility and unsupported rounds.

Primary outcomes: paired clean utility and macro/lower-decile client utility;
benign-update rejection; low/high-support approval gap; time below eligibility
threshold; committee availability. Estimate the distribution-by-trust interaction
at the seed level. Correlated votes are not independent experimental replicates.
This 2x2 design can distinguish immediate scoring harm from feedback amplification;
V9 clipping changed both filtering and trust and cannot do that alone.

Stop after the fixed batch and inspect it. If fixing trust has no benefit, do not
write that feedback caused the damage. If representative validation helps but
private validation does not, the paper must retain the data-access limitation.
If neither intervention explains the failure, audit model convergence and the
selection rule before designing another defense.

### B. Select the contribution before broad confirmation

If A identifies a mechanism, choose either one explicitly defined mitigation
or a failure-boundary study. Do not run an open-ended threshold search. Any new
mitigation first needs the existing clean-harm gate, attack-effectiveness checks,
and comparison against FedVal and clipping. Gate passage is a screening result,
not a fairness or security certificate.

For a successful mitigation, a proposed focused confirmation matrix is:
2 datasets x 5 methods x 3 conditions x 5 fresh seeds = **150 runs**.
Methods: FedAvg, clipping-only, original rank/trust, audited FedVal and the frozen
mitigation. Conditions: clean, gradual poisoning and targeted model replacement.
Use 50 enrolled clients, 40% participation and a fixed 100-round horizon after
development convergence checks. Keep FEMNIST writer partitions; use
Fashion-MNIST for controlled heterogeneity. CIFAR-10 is the first optional
transfer check, not a substitute for resolving FEMNIST.

For a failure-boundary paper, replace the mitigation row with the prespecified
mechanistic ablation and report the harmful configurations as diagnostic
comparisons, not deployable defenses. Such a new diagnostic protocol must not
retroactively release or relabel previously held V8/V9 cells.

Fresh seeds must be selected after auditing exposure across all prior studies.
Seeds 11/17/23 are development-exposed; the original five-seed group
13/29/47/71/101 must not simply be relabeled untouched. Fresh RNG seeds do not
undo tuning on reused test data: verify held-out evaluation access and disclose
reuse. Freeze endpoint definitions, primary comparisons and meaningful effect
sizes before evaluation. Report per-seed points and paired uncertainty; five
seeds do not automatically supply adequate power for a small effect.

Baseline resource accounting must distinguish unique reference examples,
class/writer coverage, validation evaluations, reference-gradient computation
and runtime. Include both the native assumptions and a separately labeled
budget-matched sensitivity. Do not cripple FedVal to manufacture a win or
describe keeping data local as a formal privacy guarantee.

### C. Stress the claimed boundary, not every possible parameter

After selecting a contribution, prioritize three targeted tests:

- Honest drift and stronger heterogeneity: does the system preserve minority
  clients when validation distributions shift without an attack?
- Independent validator corruption: false approval and withholding with poisoned
  clients controlled separately; measure acceptance, wrongful exclusion, actual
  corrupt committee counts and liveness. Population fractions alone are inadequate.
- Adaptive targeted poisoning: tune on development proxy data against the known
  validation rule, freeze attacker budget/knowledge, and evaluate unseen seeds.
  Clean utility and ASR must both be reported.

Add 20/50/80-client scale measurements on a small, justified subset if scalability
remains a claim. Record validation work, bytes and isolated wall time. Benchmark
the final configuration before forecasting; V9's roughly 29-minute pilot does
not establish the cost of 100-round confirmation or larger datasets.

### D. Recovery and blockchain are separate evidence questions

Promote recovery only if it improves on both continued training and plain
checkpoint restoration at matched detection time, suspect information and
additional compute. Measure residual ASR, accumulated damage and time to regain
utility. Separate scheduled/oracle-trigger recovery from end-to-end detection.
V4/V5 had no recovery events; V6's online recovery added no measured benefit.
Honest approval does not establish culpability, so leave poisoning-related
monetary penalties outside the main claim until attribution is supported.

For enforcement, replay the same trace in memory and a newly verified EVM
deployment. Compare state agreement and incremental gas/latency; state the
trusted registrar and eligibility-snapshot assumptions. A signed centralized
log is a meaningful alternative and has not been ruled out. Local EVM execution
does not establish distributed finality, economic deterrence or truthful votes.

## Schedule and submission decision

- October 20-21: run the bounded mechanism study, audit FedVal/FLTrust fidelity,
  and establish a held-out evaluation plan. Preserve all prior negative results.
- October 22-23: choose mitigation versus failure-boundary contribution; freeze
  the selected protocol, benchmark runtime and commit to a feasible matrix.
- October 24: abstract registration using only supported claims and clearly
  identified research objectives, within the portal's confirmed cutoff.
- October 24-28: complete the chosen confirmatory groups and essential stress
  tests; retain failed/held conditions in the completeness report.
- October 29-30: regenerate the manuscript for one protocol, verify claims,
  bibliography, figures, anonymity and nine-page content budget.
- October 31: submit only if the central claim has completed supporting evidence;
  the paper deadline is not a reason to imply that pending work succeeded.

Main-paper evidence should concentrate on four figures: clean harm/attack
tradeoff; the validation-by-trust interaction; exclusion/eligibility trajectories;
and independent validator/adaptive-attack stress. A compact cost table supports
the ledger claim. Keep SIFT1M as a small supporting case study only if it serves
the selected thesis. Recovery, EMH routing and Sybil enrollment should not each
compete to be a separate main contribution within nine pages.

The present evidence supports a serious redesign decision and a precise next
experiment. It does not yet support a general robustness, recovery, fairness,
economic-accountability or acceptance claim.
