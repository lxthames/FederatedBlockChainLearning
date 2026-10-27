# Validation and experiment-design audit - 20 October 2026

Scope: frozen V9 validation/aggregation/evaluation code, the V8 pairing check
reused by V9, and the proposed 36-cell mechanism study. This is a bounded review,
not a full security or baseline-fidelity audit. Experimental code and completed
study artifacts were not modified. Findings distinguish implementation defects,
method limitations, and limitations of the proposed causal interpretation.

Reproduction command:

```powershell
.venv-trust/Scripts/python.exe -m scripts.audits.audit_validation_strategy_20261020
```

The script refuses to overwrite prior audit output. Its completed evidence is
`artifacts/validation-strategy-audit-20261020/report.json`, with an isolated
four-round CPU trace under `withholding/`. Input source hashes are recorded.

## 1. High: missing quorum penalizes the proposer as if its update were rejected for quality

Locations: `scripts/trust_bias_v9/research/runner_v2.py:263`, `:278`;
`scripts/trust_bias_v9/research/defenses.py:204`, `:227`.

Withheld votes are omitted, but failure to achieve the approval quorum produces
the same false acceptance flag as a substantive rejection. Aggregation reduces
the proposer's trust for that flag. The decision has no separate insufficient-
response outcome. Thus validator nonresponse can damage another client's trust.

Reproduced end to end with four clients, full participation, two withholding
validators, **zero poisoned clients**, and keep fraction 1 so all received votes
approve. After three rounds:

- Withholding validators 0 and 1 have trust 0.56; their proposals are approved
  by the two honest validators.
- Honest validators 2 and 3 have trust 0.44, below the 0.45 eligibility threshold.
  Their proposals lack sufficient responses; there are no negative votes.
- In round four no candidate has enough eligible validators to form a committee.

This small construction does not establish failure frequency at 50 clients or
violate a claimed honest-quorum guarantee: the committees here lack the required
responsive honest quorum for some candidates. It establishes a separate
accounting failure: missing evidence is used to penalize an honest proposer,
and feedback worsens availability. Current V9 efficacy runs used zero corrupted
validators, so this is not evidence that withholding caused their clean losses.

Required change in a new version: separate accepted, quality-rejected and
insufficient-response outcomes. Missing quorum must not itself reduce proposer
trust. Report observed response/approval counts, reasons for nonacceptance,
and committee availability separately. A missing quorum alone also does not
prove validator malice in a real deployment with network failures.

## 2. High for the proposed study: changing validation partitions changes the fairness measurement

Location: `scripts/trust_bias_v9/research/splits.py:68`, especially `:79`;
consumer: `scripts/trust_bias_v9/research/runner_v2.py:494`.

For standard datasets, held-out per-client test groups are synthesized from each
client's **training plus validation** class masses. The proposed diagnostic
redistributes validation examples while holding training fixed. With this helper,
that intervention changes the test-group membership used to compute worst-client
and lower-decile utility, even when predictions are identical.

Reproduced with two fixed training clients, the same 80 validation examples,
the same 200 test examples and constant class-0 predictions:

| Metric | Original private validation | Redistributed validation |
|---|---:|---:|
| Global accuracy | 50.00% | 50.00% |
| Reported worst-client accuracy | 0.00% | 43.56% |
| Reported lower-decile client accuracy | 10.00% | 44.86% |

The model did not improve. This is a measurement confound for the proposed
intervention, not evidence that completed V9 global accuracy is wrong. The
natural-client FEMNIST branch uses explicit writer test groups and avoids this
specific dependency.

Required change: precompute and hash the test-group assignment before intervention
and reuse it in every arm. Use fixed writer groups for FEMNIST. For Fashion-MNIST,
declare the groups synthetic and derive them from a fixed original partition or
training-only distribution. Keep client IDs and group sample counts in outputs.
Test that validation redistribution cannot alter evaluation-group membership.

## 3. Medium: rank acceptance is relative selection, not evidence of safety or misconduct

Location: `scripts/trust_bias_v9/loss_voting.py:90` to `:105`;
trust coupling: `scripts/trust_bias_v9/research/defenses.py:227`.

The rank rule selects the top fraction without an absolute loss-benefit condition.
Using the actual LossCommittee rule with five equal-norm proposals and a separate
validator, the following mocked paired losses both produce approvals
`[true, true, true, true, false]`:

- Every update helpful: loss changes -0.5, -0.4, -0.3, -0.2, -0.1.
- Every update harmful: loss changes +0.1, +0.2, +0.3, +0.4, +0.5.

The demonstration sets the norm penalty to zero to isolate ranking; equal norms
also contribute the same penalty and leave the ordering unchanged. This is an
intentional property of the implemented rule, not a sorting bug. Equal-score
ties and ceiling rounding mean it does not always reject exactly 20%.

It matters because a rank rejection reduces trust and can remove voting
eligibility. Relative performance on one validator's data is not culpability.
The V9 uncertainty-backoff variant can admit extra proposals, but its clean
failure shows that this particular attempted remedy is not established.

Required next evidence: separate rank rejection from measured local loss harm,
report both, and test fixed versus dynamic trust. Do not add an unvalidated
absolute threshold and call the problem solved: local/global objective mismatch
persists. Keep monetary poisoning penalties disabled unless attribution is justified.

## 4. Medium: the paired-comparison check does not bind client partitions

Location: `scripts/data_preparation/run_comparison_v8.py:113`, called by
`scripts/trust_bias_v9/study.py:120`.

`matched` compares participant schedules and full training/test array fingerprints.
It does not compare allocation of training examples to clients, initialization,
or per-client evaluation groups. Identical full arrays do not imply identical
client partitions.

A synthetic metadata reproduction gave two rows different training-split digests
while retaining those three existing fields; `paired_row` still reported matched.
This establishes a validation gap, not an observed historical split mismatch.
The frozen coordinators provide additional provenance checks for prior studies.

Required change: bind and compare training partitions, fixed evaluation groups,
initial-model hash and common training settings. Bind each validation partition
separately and verify the prespecified redistribution (same union, no duplicates,
same per-validator counts); validation hashes should intentionally differ between
those arms. Requiring every split hash to be identical would reject the experiment
we actually intend to run.

## 5. Medium design limitation: switching trust off changes two causal channels

Locations: `scripts/trust_bias_v9/research/defenses.py:211` and
`scripts/trust_bias_v9/research/runner_v2.py:203`.

Trust controls both aggregation weights and validator eligibility. Therefore the
proposed dynamic-versus-fixed comparison measures their combined effect. If
fixed trust improves utility, that alone does not show that lost committee
eligibility caused the original damage; reweighting could explain it.

The existing V9 clipping-only control changes still more: it sets rank keep
fraction to 1 and reward/rejection updates to zero simultaneously
(`scripts/trust_bias_v9/study.py:35`). Its clean improvement cannot separately
attribute harm to filtering or trust.

For the 36-cell study, explicitly label the trust contrast as the total effect
of trust weighting and eligibility. If the paper claims eligibility feedback,
add a targeted arm that retains dynamic trust weighting but holds validator
eligibility to its baseline population. On the original private validation
condition, two datasets times three seeds add six logical cells (42 total).
This is a targeted channel intervention, not a complete factorial estimate of
all interactions. It needs a frozen new protocol before execution.

## Verification and next action

- The reproduction script completed all four assertions: withholding, moving
  evaluation groups, relative ranking and incomplete pairing checks.
- Existing V9 tests: **11 passed** (`pytest tests/test_bias_v9.py -q`). Passing
  these tests does not cover the reproduced failure cases.
- The current V9 package source hashes still equal its frozen source manifest.
- No full training batch, mitigation or new confirmatory run was launched.

Repair the quorum-outcome distinction and freeze evaluation groups/matching
before running the diagnostic study in a separately versioned implementation.
Preserve original rank/trust behavior as an explicit comparison arm. Treat the
relative-ranking and trust-channel findings as research questions; they require
evidence beyond implementation repairs. Do not silently regenerate historical
results using a changed protocol or evaluation definition.
