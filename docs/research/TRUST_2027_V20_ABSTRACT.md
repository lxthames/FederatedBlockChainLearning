# Updated registration draft after completed V20

Prepared 21 October 2026. Local draft only; CMT registration deferred by user.
This supersedes the V19-only abstract in the earlier submission-plan document.
Final authorship, track selection, novelty review and submission metadata remain
pending. Do not present the checkpoint scope as an online defense evaluation.

**Title:** Teacher-Anchored Backdoor Repair under Heterogeneous Private Validation

**Abstract:** Private validation in federated learning must assess model repairs
using limited, heterogeneous local data. A repair that reduces attack success
may be rejected, making its raw efficacy differ from the efficacy of the model
ultimately admitted. We study this distinction using matched FEMNIST checkpoint
experiments and evaluate a teacher-anchored patch repair that freezes
convolutions, aligns clean and patched outputs with a frozen parent model on
clean inputs, and penalizes non-label logit increases. In a 24-cell diagnostic
on six round-100 development checkpoints, we reproduce an interleaved supervised
repair baseline and compare three anchored objectives under unchanged disjoint
private validation panels. The full candidate reduces mean committed trigger
target rate from 69.21% to 9.23%, with per-seed rates of 10.31%, 7.78%, and 9.61%.
All three clean and three poisoned repairs are admitted. Mean clean accuracy is
77.59%, compared with 78.54% for the clean source checkpoints, and every seed
satisfies the declared utility limits. The ablations distinguish clean utility
preservation from backdoor removal and expose residual admission costs. We
report exact baseline reproduction, raw and committed metrics, client-level
utility, and additional teacher computation and communication. These results
establish checkpoint-level feasibility within exposed development data and a
known patch bank; they do not yet establish online resilience, unseen-trigger
generalization, or independent confirmation.

**Keywords:** federated learning; backdoor repair; private validation; knowledge
distillation; heterogeneous data; reproducible evaluation.

## Submission decision

The [official CFP](https://eigtrust.acm.org/trust2027/cfp/) lists abstract
registration October 24 and full paper submission October 31, 2026. This draft
fits the secure/resilient AI and evaluation topics, but the correct CMT track
must be selected from the actual submission options. Its novelty and evidence
scope still need author review; no acceptance-readiness claim is made.

For a paper claiming a new federated defense, add separately frozen end-to-end
and independent evidence before submission. If those studies cannot finish,
the paper must retain the explicit checkpoint scope and establish a specific
novel empirical contribution; changing the title does not resolve that burden.
