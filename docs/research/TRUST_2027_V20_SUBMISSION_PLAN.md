# TRUST 2027 submission plan after V19

Prepared 21 October 2026. Abstract registration is deferred at the user's request;
no CMT submission or external message has been sent. This is a local working plan.

## Verified deadline and scope

The [official CFP](https://eigtrust.acm.org/trust2027/cfp/) lists abstract
registration October 24, 2026 and full paper submission October 31, 2026.
It requires double-blind ACM sigconf, at most nine pages including appendices;
references and required GenAI Usage Disclosure may use additional pages.
The strongest fit for the evaluation article is Symposium 5, Test and Evaluation,
Benchmarking & Assurance, whose official CFP provides [CMT Track 10](https://cmt3.research.microsoft.com/TRUST2027/Track/10/Submission/Create).
If a validated defense becomes the primary contribution, Symposium 2, Secure and
Resilient AI Systems, provides [CMT Track 7](https://cmt3.research.microsoft.com/TRUST2027/Track/7/Submission/Create).
The cutoff timezone remains unverified. Author details and account access remain
to be supplied later.

The [workshop call](https://eigtrust.acm.org/trust2027/call-for-workshop-proposals/)
is for workshop organizers; it does not establish the pasted December paper
deadline or guarantee proceedings inclusion for a selected workshop. A workshop
fallback requires that workshop's actual CFP. Do not treat it as secured.

## Decision schedule

| Date | Deliverable | Decision |
|---|---|---|
| Oct 21-22 | Freeze/run 24 V20 checkpoint cells; exact V19 controls; audit all outputs | Apply declared gate; no outcome-guided retuning |
| Oct 22-23 | Novelty table against private validation, distillation and federated hardening; finalize claim | Distinct mechanism or demonstrated defense improvement must be stated precisely |
| Before Oct 24 cutoff | Authors review title/abstract and complete CMT metadata | Register an evidence-backed claim, with no invented V20 outcomes |
| Oct 24-28 | If V20 qualifies, separately freeze end-to-end and independent confirmation; otherwise assess focused failure study | A checkpoint pass alone cannot support an online defense claim |
| Oct 29-30 | Reconcile every claim/figure with sealed results; anonymity, formatting, references, GenAI disclosure | Remove unconfirmed claims; resolve missing evidence or defer submission |
| Before Oct 31 cutoff | Submit only a complete supported paper | No promise that the remaining experimental requirements fit the deadline |

## Current claim and contribution boundary

Supported development observation: under this FEMNIST protocol, raw repair
efficacy and admitted repair efficacy differ. Scheduling can improve admission
while worsening average raw trigger removal. Clean private-panel decisions can
also disagree with global utility. These are measured results, not a universal
verification-repair impossibility theorem or a new defense guarantee.

V19 interleaving changes hard-patch raw trigger rate 47.45% to 50.50% while
committed trigger rate changes 84.01% to 69.21%, with attacked admission 1/3 to
2/3. Mean clean raw accuracy improves 74.14% to 74.92%, still below its 78.54%
source. Interleaved clean-only tuning reaches 75.79%. Three exposed seeds,
one architecture, fixed late checkpoints, one attack, and an evaluation trigger
inside the challenge bank substantially limit generalization.

An arithmetic decomposition is possible but must not be presented as causal:
source minus sequential hard = (source minus interleaved clean-tune) +
(interleaved clean-tune minus interleaved hard) +
(interleaved hard minus sequential hard), approximately
4.395pp = 2.752pp + .867pp + .775pp. The first contrast does not isolate sample
size and the second does not isolate occlusion from optimization. There is no
evidence that a stated percentage of harm is caused specifically by missing data.

For V20, the possible claim is conditional: a frozen-teacher repair objective
can improve BOTH raw and admitted backdoor removal at comparable utility.
Do not write that claim as achieved until the frozen gate and stronger studies
support it. Known KL/freeze/suppression components alone are insufficient novelty.

## Nine-page working allocation

| Content | Pages | Evidence obligation |
|---|---:|---|
| Problem, contribution and related work | 1.5 | Specific distinction from CrowdGuard, DifFense, FLIP, prior repair/distillation |
| Threat model, private validation and measurement protocol | 1.5 | Honest repair-provider assumption; empirical panels; raw/committed definitions |
| Matched diagnostic and mechanism limits | 1.5 | V19 2x2; distribution of effects; no unsupported causal decomposition |
| V20 rule and ablations, if useful | 1.0 | Exact losses, combined-change confounds, clean teacher poisoning risk |
| Results and generalization | 2.5 | Utility, attacks, lower tail, holds, overhead, independent confirmation |
| Limitations and conclusion | 1.0 | Exposure, in-bank trigger, missing online evidence, no formal privacy |

This allocation is provisional. If V20 fails, reduce the algorithm section and
require a stronger generalizable failure mechanism; failed variants alone do
not constitute an evaluation contribution. Do not expand discussion to hide a
missing novelty or generalization result.

## Response to tracked Globecom issues

The local reviewer tracker is the source of these issue labels, not newly
recovered verbatim reviews.

| Concern | Current response | Still required |
|---|---|---|
| R3 novelty | Precise raw-versus-committed question; controlled scheduling contrast | Closest-work distinction, mechanism beyond known validation limitations |
| R4 heterogeneous validation bias | Disjoint repair/screen/verify clients; private vs global outcomes explicit | Support/heterogeneity intervention and lower-tail confirmation |
| R2 exaggerated robustness | Preserve failed arms, holds and raw/committed trigger rates | No claimed success from collapse or evaluation-aware selection |
| R1/R3/R4 limited evaluation | Frozen 100-round source trajectories and matched checkpoint studies | Online repeated attack, gradual utility, independent data/seeds and attacks |
| R2 incomparable baselines | Exact V19 reproduction with extra teacher costs reported | Fidelity-audited FedVal, FLAME, CrowdGuard and applicable repair comparator |
| Blockchain/accountability claims | Hash equality and completion seals verify artifact integrity | These hashes do not establish validator honesty or poisoning attribution |

## Abstract draft (current measured evidence only)

**Working title:** From Raw Repair to Admitted Repair: Evaluating Backdoor
Mitigation under Heterogeneous Private Validation

Private validation in federated learning must assess updates and repairs using
limited, heterogeneous local data. We study the distinction between a repair's
measured efficacy and the model ultimately admitted by private validators. In
three FEMNIST development seeds, a clean-loss quorum produces round-100 models
with a mean non-target trigger target rate of 99.96% under model replacement.
A matched 24-cell checkpoint diagnostic crosses sequential versus interleaved
repair with clean-only versus hard-patch supervision while preserving example
and update budgets. Interleaving improves hard-patch clean accuracy from 74.14%
to 74.92%, compared with 78.54% for the clean source checkpoints. On poisoned
checkpoints, mean raw trigger target rate worsens from 47.45% to 50.50%, yet
committed trigger target rate improves from 84.01% to 69.21% as more repairs pass
the unchanged private verifier. These results show that admission can change
the apparent ranking of repair strategies. We report raw and committed utility,
private decisions, and computation and modeled communication costs with verified
checkpoint provenance. The findings are limited to exposed development seeds
and an in-bank trigger; they do not establish a general defense or independent
confirmation. We identify the evidence needed to distinguish useful repair
from improvements caused primarily by admission changes.

This draft deliberately excludes unmeasured V20 outcomes. It must be revised
after the novelty audit and final scope decision; its existence is not an
assessment of acceptance readiness. No abstract has been registered.

## GenAI disclosure draft for author review

Generative AI assistance was used to help implement experimental tooling,
design software checks, analyze recorded outputs, and draft research text.
The final disclosure must identify the actual tools/models and contributions
used across the project. Authors remain responsible for verifying methods,
results, citations and claims. Do not claim human checks that have not occurred.
