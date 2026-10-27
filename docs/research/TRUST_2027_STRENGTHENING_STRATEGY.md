# TRUST 2027: literature, dataset and reviewer-risk strategy

Prepared 19 October 2026. The initial assessment is followed by the versioned
implementation documented in [the extension runbook](TRUST_EXTENSION_RUNBOOK.md).
FedVal, natural-client FEMNIST, adaptive and validator stress tests, recovery
controls and trace agreement checks are implemented; the new background study
has 33 development cells and up to 275 evaluation cells. Experimental resolution
remains pending. Existing jobs continue under their recorded configurations.

## Recommendation and venue fit

Complete the existing Fashion-MNIST/CIFAR-10 evaluation and verified SIFT1M case
study. Strengthen the comparison with the closest validation and recovery work,
then test the mechanisms most likely to fail. Add one dataset only if it answers
a distinct question and complete seed groups can finish before the analysis gate.

The paper fits the official CFP's Secure and Resilient AI Systems topics
(poisoning, federated learning and runtime monitoring), with supporting relevance
to Evaluation, Benchmarking and Assurance. This is a scope assessment, not a
prediction of acceptance or a verified CMT track selection. The venue explicitly
requests supported claims, clear methodology and limitations. Its main-paper
budget is nine pages including appendices; references and GenAI disclosure may
extend beyond that budget. See the [official CFP](https://eigtrust.acm.org/trust2027/cfp/).

The strongest defensible contribution would be evidence about the tradeoff
between recovery benefit, mistaken sanctions, committee availability and
enforcement cost under explicit trust assumptions. Integration alone remains
vulnerable to the earlier novelty criticism. New datasets cannot establish
algorithmic novelty.

## Priorities

| Priority | Work | Completion evidence |
|---|---|---|
| P0 | Finish core seed groups; classify failed/held cells; verify attack and baseline fidelity | Immutable artifacts, per-cell completion table, matched clean controls, audited assumptions |
| P0 | Compare recovery with continuing training and plain checkpoint restoration | Paired utility/ASR trajectories, residual damage, training work and time to regain utility |
| P0 | Test adaptive evasion and validator-driven denial of service | Frozen attacker knowledge/budget, attack validation, wrongful penalties and unsupported rounds |
| P0 | Explain ledger necessity and its limits | Same trace in memory and EVM; state agreement; actual gas and latency; explicit trusted-operator boundary |
| P1 | Add closest validation baseline, FedVal, after source audit | Matched reference budget and training configuration; complete five-seed comparisons |
| P1 | Add natural-client FEMNIST study if runtime permits | Verified writer partitions, locked protocol, complete focused matrix |
| P2 | Add a security-domain dataset or additional defenses | Only when P0/P1 evidence is secure and the addition answers a new question |

## Literature revision

Use the accompanying [literature assessment](TRUST_2027_LITERATURE_ASSESSMENT.md).
Organize the review around four questions: who evaluates updates, what information
the evaluator needs, what recovery actually guarantees, and who enforces the
recorded decision. Compare assumptions and mechanisms in a compact table; keep
cross-paper accuracy and gas figures out of a common ranking.

FedVal is a high-value addition because validation and heterogeneous-client
fairness are central to this paper. FLDetector addresses historical detection;
FLAME covers targeted backdoor defense. FedRecover and Crab are essential recovery
comparators. Retain the recent VerifBFL and FAUN discussion, with the latter
explicitly identified as a preprint. The verified ICLR 2026 data-poisoning work
provides recent theoretical context, but is not automatically a compatible
arbitrary-model-update baseline. Recency alone is not a reason to implement a
method.

Before adding numeric comparisons, audit full papers and author code, pin the
source revision, reproduce algorithmic operations and document incompatible
assumptions. This review verified primary descriptions and metadata, not full
reproductions of these additional methods.

## Dataset decision

| Dataset | Distinct question | Recommendation |
|---|---|---|
| Fashion-MNIST | Controlled mechanism, drift and parameter studies | Finish current study |
| CIFAR-10 | More complex visual task under the same evaluation protocol | Finish current study |
| SIFT1M | Does the framework transfer to routing, and does EMH-DNN help under a matched routing budget? | Keep as a supporting case study; use only new verified labels |
| FEMNIST from LEAF | Do natural writer-specific distributions cause biased votes, exclusions or sanctions? | First additional dataset, conditional on feasibility |
| ToN-IoT network data | Does the approach work on tabular security data with a credible deployment partition? | Optional later extension; requires a separate leakage audit |
| MNIST, another similar easy image benchmark | Mostly repeats current evidence | Low priority |
| CIFAR-100, large language or billion-vector tasks | Increased task/model scale with substantial new constraints | Defer for this revision |

[LEAF](https://github.com/TalwalkarLab/leaf) provides FEMNIST with natural
writer clients and 62 classes. Preserve writer identity rather than applying
Dirichlet repartitioning. Prespecify writer selection and minimum sample rules
without examining test performance. For selected writers, reserve disjoint
training, private validation and held-out examples; use separate writers for
reference/development data where practical, recording the resulting reference
distribution mismatch. Hash raw files, transformations, writer IDs and splits.
Fit preprocessing only on training data. Report sample-weighted accuracy,
macro-client accuracy, lower-decile client accuracy, worst-client accuracy with
sample counts, clean-update rejection, exclusions and sanctions by client group.
Writer variation is not evidence of demographic fairness.

Suggested focused FEMNIST matrix: 50 enrolled writers, 40% participation,
100 rounds, one local epoch; five seeds 13/29/47/71/101; clean, targeted model
replacement and gradual poisoning; FedAvg, FLTrust, audited FedVal, validation-only
framework and full framework. This is 75 evaluation runs, plus development runs
on 3/5/7. Use equal reference pools where required and matched client splits.
Determine model and learning rate from development convergence, then freeze.
Check attack effectiveness on this dataset rather than inheriting a gate from
Fashion-MNIST. An incomplete method must be reported, never silently replaced.

Do not blindly carry BOBA into full 62-class FEMNIST: its class-subspace support
requirements must be checked against the number of participating updates. The
default 20 participants may be inadequate for a 61-dimensional affine subspace.
Likewise, writer-level label coverage can invalidate reference assumptions.
Report applicability limits or design a separately justified configuration.

[ToN-IoT](https://research.unsw.edu.au/projects/toniot-datasets) supplies network,
telemetry and OS data, including security-event provenance. A network-only study
would need a common feature schema and credible host/site clients. Audit duplicate
flows and overlap, split by appropriate time/session units, remove label proxies
such as scenario-identifying fields, and fit scaling on training only. Distinguish
attack traffic labels from malicious FL participants. Do not invent 50 independent
devices by randomly partitioning rows; synthetic clients must be labeled as such.
Use macro-F1, per-attack recall, false-alarm rate and class prevalence alongside
accuracy. Its official source permits academic use and specifies citations.

## Anticipated objections and required responses

These are our risk assessments, not known comments from TRUST reviewers.

| Likely objection | Concrete response | Evidence needed / claim limit |
|---|---|---|
| The method combines known components | State a systems contribution and test interactions between validation, recovery and sanctions | Full vs validation-only vs recovery without penalties, matched histories and uncertainty; do not claim a new fundamental algorithm |
| Why use blockchain if the orchestrator is trusted? | Separate audit/enforcement from learning quality; compare identical decisions with and without EVM execution | Transaction/state agreement and incremental cost; a signed centralized log remains a plausible alternative, not experimentally ruled out |
| Checkpoint rollback is already known | Compare continuing training, plain rollback and the accountable version; add one compatible recovery method if feasible | Same detection time and information, residual ASR and recovery cost; no certified-clean-state claim |
| Recovery baselines receive an unfair oracle | Separate end-to-end detection from recovery-only experiments | Fix trigger time and available suspect list across recovery methods; label any oracle-suspect experiment explicitly |
| Adaptive attacks evade aggregate accuracy monitoring | Add a development-tuned attack constrained to preserve clean utility and evade observed validation thresholds | Only public decisions or attacker-owned proxy data unless explicitly omniscient; targeted ASR, cumulative loss and delay |
| Corrupt validators can frame honest signers | Exercise false approvals, false rejections and false recovery attestations, independently of poisoned clients | Wrongful penalties, retained honest stake, eligibility, quorum failures and liveness |
| Sanctions improve accuracy by excluding difficult honest clients | Inspect which clients lose stake/eligibility under clean heterogeneous data | Lower-tail utility, minority-label performance and exclusion rate; no unqualified fairness claim |
| Two-thirds quorum guarantees quality | State the conditional protocol guarantee | Membership and counting integrity do not imply truthful numerical assessments or honest-majority committees |
| The detector only finds attacks that reduce average accuracy | Evaluate backdoors that retain clean accuracy and benign drift that lowers it | Separate targeted ASR from utility; report misses and false alarms, including detection-free attacks |
| Extra data or compute makes comparisons unfair | Report reference/private validation counts, per-round scoring work and training budget | Common held-out tests; reference-size/mismatch sensitivity; isolated runtime subset |
| Controlled attacker participation hides risk | Include a separately labeled uniform-participation sensitivity | Sampled attacker counts and violated baseline bounds; failures retained, not counted as robustness |
| Results are selected from successful cells | Publish a completeness table and attack gates | Complete seed groups, all failed/held conditions, intervals and paired effects; no complete-case superiority claim hiding failed groups |
| Local EVM proves decentralized scalability | Limit claims to local contract execution | Real receipts and complete traces; no distributed finality, adversarial consensus or economic-deterrence conclusion |

For uniform sampling without replacement, a useful conditional analysis is
P(X >= q) = sum from j=q to min(M,C) of choose(M,j) choose(E-M,C-j) / choose(E,C),
where E excludes the proposer, M is the number of corrupt eligible validators,
C is committee size and q=ceil(2C/3). This describes a malicious quorum under
the stated sampling model, not all ways of accepting a bad update. Honest
validation can also err. Blocking a unanimously approving honest set requires
more than C-q withholding/rejecting validators. Eligibility changes make E and M
time dependent; use round-specific counts and do not assume independent windows.
Prove only the implemented contract invariants, not poisoning detection itself.

## Implementation and scheduling safeguards

The active runners fingerprint scripts/research. Additions must use a separately
versioned experiment snapshot and new output root, or begin after the current
study finishes. Never modify those sources in place while the jobs run. Reuse
existing results only when source/config/data compatibility is established.

The current ALIE adapter uses a fixed z=1 mean/std rule and privileged honest-peer
reference updates. Audit it against the original threat model and parameter
selection before calling it a faithful standard ALIE reproduction. Two observed
development failures involve too few reference peers after eligibility changes.
This calls for an explicit attacker-information/availability policy and separate
versioned reruns if changed, not an undocumented fallback to benign updates.

Before expanding: (1) inspect completed development convergence and gate results;
(2) benchmark representative clean, attacked and full-framework runs on the
actual device; (3) forecast remaining core, sensitivity and optional hours with
headroom; (4) freeze the new manifest before its confirmatory runs. Use new
development seeds for changes motivated by already observed confirmatory results,
and disclose that an extension is a follow-up study. Do not keep tuning until a
desired ranking appears.

By internal October 23 Taipei: settle the contribution, audit closest baselines,
fix attack applicability and decide FEMNIST feasibility. Abstract registration is
October 24; use only supported claims or clearly stated research objectives.
By October 28: close the intended evaluation matrices for analysis. By October 30:
complete claims, page budget, anonymity and visual checks. Verify the portal's
exact cutoff independently. Essential missing evidence keeps the work a draft;
dropping an optional dataset is preferable to representing partial groups as
complete evidence.

Suggested nine-page allocation: introduction 0.75, related work/comparison 0.75,
threat model 0.75, protocol 1.5, prototype 0.75, evaluation 3.5,
limitations/conclusion 0.5, with 0.5 page layout margin. This is a planning budget,
not the current rendered length. SIFT should occupy a compact supporting result.
