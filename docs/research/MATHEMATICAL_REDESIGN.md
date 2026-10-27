# Literature-guided mathematical screening (19 October 2026)

Status: experimental alternatives, not a validated final protocol. Code lives in
`scripts/trust_math_lab/`. Original, extension and v3 snapshots remain unchanged.

## What the failed runs establish

The three v3 FEMNIST clean runs accepted 6.5%, 14.2%, and 3.05% of submitted
updates overall, with 80, 68, and 92 no-update/unsupported rounds respectively.
Each had false recovery events (1, 2, 1), imposing 90, 250, and 235 stake credits
of penalties on honest validators. These are development diagnostics, not a
validated attack-defense result. The overall acceptance fraction includes
unsupported rounds and must not be mistaken for a per-vote approval probability.

For F(w) = sum_k p_k F_k(w), a local update Delta_k approximately follows
-eta grad F_k(w), but its first-order effect at validator j is

    F_j(w + Delta_k) - F_j(w) ≈ -eta <grad F_j(w), grad F_k(w)>.

The inner product can be negative under heterogeneous labels. Local improvement
for every validator is not necessary for improvement of the weighted global
objective. Hard candidate filtering can therefore select clients by similarity
to validators rather than maliciousness. SCAFFOLD analyzes client drift;
BOBA specifically identifies robust-aggregation selection bias under label skew.

Even if each independent validator approved a benign candidate with probability
1/2, C=15 and q=10 yield only about 0.151 probability of committee approval.
This is an illustrative binomial calculation, not a claim that actual votes are
independent. Shared model errors and writer distributions correlate votes.

For approval reward r and rejection penalty d, expected trust change is
p*r - (1-p)*d. With r=d=.02, any approval probability below .5 decreases trust.
Coupling this to validator eligibility can turn repeated benign disagreement
into loss of committee availability. Increasing learning rate cannot fix that
feedback loop. Initial screening freezes trust and disables recovery to isolate
the voting rule; these ablations do not validate a full recovery system.

## Primary literature and limits of transfer

- [Zeno, ICML 2019](https://proceedings.mlr.press/v97/xie19b.html): ranks gradients
  by validation-loss descent minus a squared-norm term. Its analysis assumes an
  unbiased loss oracle, with evaluation samples drawn after update arrival.
  Fixed private writer validation does not satisfy those assumptions, so we
  transfer a scoring idea, not the theorem or a claim of faithful reproduction.
- [FedVal, USENIX Security 2023](https://www.usenix.org/conference/usenixsecurity23/presentation/valadi):
  uses server validation-loss information for aggregation. It motivates moving
  away from discontinuous accuracy alone; our committee rules are not FedVal.
- [SCAFFOLD, ICML 2020](https://proceedings.mlr.press/v119/karimireddy20a.html):
  corrects client drift using control variates. It supports investigating
  heterogeneity, but SCAFFOLD is not implemented in this screening and does not
  itself provide the Byzantine security needed here.
- [BOBA, AISTATS 2024](https://proceedings.mlr.press/v238/bao24a.html): addresses
  label-skew selection bias and robust affine-subspace estimation. Its geometric
  assumptions must still be checked at 62 classes and 20 participants.
- [Fixing by Mixing, AISTATS 2023](https://proceedings.mlr.press/v206/allouah23a.html):
  studies robust learning under heterogeneity using nearest-neighbor mixing.
  This is a further geometry-based option if private validation remains biased;
  it is not claimed as an implemented baseline here.

## Three mathematical alternatives

Let ell_i(w) be cross-entropy on private validation example i. No official test
arrays are read. Let d_i = ell_i(w + Delta_k) - ell_i(w).

**Paired loss rule.** Calculate d_bar and SE = sd(d_i)/sqrt(n). Approve when

    d_bar <= epsilon + z * SE,   epsilon=.01, z=1.64.

Pairing cancels shared example difficulty. The rule rejects on evidence of
harm; it does NOT establish noninferiority. A formal noninferiority rule would
place the uncertainty term on the other side and would reject more uncertain
candidates. With n<2, withhold approval. This margin is exploratory: repeated
reuse, adaptive models and within-writer dependence prevent interpreting it as
a calibrated sequential confidence guarantee. No FPR guarantee is asserted.

**Contribution-scaled loss rule.** Set a_k = n_k / sum_l n_l, and evaluate
w + a_k Delta_k using the same paired rule. With fixed equal trust, aggregate

    w_next = w + sum_{k accepted} a_k Delta_k.

Rejected mass remains zero; approved updates are not renormalized upward. The
implementation checks that trust is fixed and recovery disabled for this variant.
This matches each probed coefficient to its aggregation coefficient. Nonlinear
interaction among multiple updates can still cause harm, and smaller probe
steps can hide backdoors: targeted ASR must be checked independently.

**Loss-ranking rule.** Each validator scores candidates other than its own:

    s_jk = mean_i[ell_i(w)-ell_i(w+Delta_k)]
           - .001 * (||Delta_k|| / max(median_l ||Delta_l||, 1e-12))^2.

Approve the top ceil(.8 * number_of_candidates) scores locally, including all
ties at the cutoff. The usual distinct-signer two-thirds quorum still applies.
Relative ranking avoids requiring every benign local model to improve every
writer. Conversely, ranking always prefers some candidates even if all are bad;
the .8 retention fraction is a design parameter, not a Byzantine bound.
Each participating validator needs comparison updates outside some candidate
committees, so the communication estimate includes these extra deliveries.

All three use median-based clipping before evaluation AND aggregation:

    Delta'_k = Delta_k * min(1, 2 median_l ||Delta_l|| / max(||Delta_k||,1e-12)).

Median norm stability presumes adequate honest representation and comparable
norms. Clipping limits magnitude, not poisoning direction or targeted behavior.
An accuracy-plus-clipping control separates clipping effects from the loss rule.

## Frozen bounded experiment and stopping rules

Use existing training-only FEMNIST development splits and seeds 11/17/23.
These seeds reuse development writer sets; they are not new independent datasets.
Every run: 50 clients, .4 participation, 40 rounds, one epoch, batch 16, SGD
lr .05/momentum .9, same initial model/partition/participant streams.

Stage 1: 24 clean runs (eight variants x three seeds): FedAvg; v3 full; v3
validation-only with dynamic trust; accuracy with fixed trust; accuracy plus
clipping with fixed trust; paired loss; scaled loss; loss ranking. The three
loss rules use fixed trust and validation-only behavior.

Stage 2: six FedAvg development attack controls, gradual and targeted replacement.

Stage 3: at most 18 defended attack runs, only for alternatives that pass the
complete clean group AND attacks that pass the complete FedAvg development gate.
Clean feasibility requires every baseline > majority-class accuracy + .05,
mean paired clean loss no worse than .05 and no seed worse than .10. Attack
effects must exceed .005 utility loss or .05 targeted-ASR lift. These are
prespecified heuristic development screens, not significance tests. Incomplete
or failed groups cannot pass. Held cells retain explicit reasons. No automatic
confirmation or winner selection occurs.

Run: `.venv-trust/Scripts/python.exe -m scripts.trust_math_lab.study`.
Status/analysis: `artifacts/trust-math-lab-v1/`.
Results: `results/trust-math-lab-v1/`. Source, configuration and data hashes are
frozen. Interrupted cells are preserved and restarted; completed cells require
matching configuration. No simultaneous duplicate coordinator is allowed.

A promising alternative must next survive recovery/trust reintegration, independent
development replication, Fashion-MNIST/CIFAR transfer, validator corruption,
adaptive attacks, contract trace checks and final-protocol coverage. Selecting
the best of several development alternatives creates selection bias; disclose
the search and do not present these screening runs as confirmatory evidence.
Earlier official-test exposure remains disclosed in TRUST_REVISION_V3.md.

## Final screening assessment

`scripts/data_preparation/assess_math_screening.py` reads immutable summaries and
writes `artifacts/trust-math-lab-v1/assessment.json`. Its `--watch` mode updates
that report every 30 seconds and exits when the study stops. It performs no
training and cannot launch confirmation or promote a protocol automatically.

The additional assessment policy is recorded with its source hash and explicitly
states that it was introduced after seven first-seed clean results were visible.
This is development analysis, not a preregistered confirmatory test.

A candidate is eligible for reintegration review only with complete matched
three-seed groups, passing clean learning/cost, both attacks effective against
FedAvg, and positive measured defense benefit for both attacks. Untargeted
defense needs mean utility benefit > .005; targeted defense needs mean ASR
reduction > .05 and mean utility no worse than FedAvg by .05. These are
development decision thresholds, not statistically established guarantees.
Duplicate seeds, incomplete groups, ineffective attacks and mismatched artifact
configuration hashes cannot qualify. All summaries retain their failure/held
status. A clean-accuracy gain alone cannot qualify a method that retains a strong
backdoor. Recovery/trust reintegration remains a separate experiment.
