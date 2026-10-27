# V20: bounded frozen-teacher repair diagnostic

Declared before V20 outcomes, 21 October 2026. This is a new authorized study;
V17, V18 and V19 stop decisions and all historical sources remain unchanged.

## Question and scope

Can clean teacher anchoring with frozen convolutions preserve utility, and does
paired patch distillation improve removal relative to V19 at unchanged admission?
These are hypotheses. No guaranteed admission, ASR, layer localization, or
publication novelty follows from the proposed loss.

Six sealed round-100 FEMNIST checkpoints: clean and model replacement for exposed
seeds 11, 17, 23. Four arms per checkpoint, 24 cells total. This is a nested
ablation, not a full factorial of freezing, anchoring, invariance and suppression.
Each checkpoint reproduces the V19 interleaved hard-patch baseline first. All
200 optimizer steps, selections, model hashes, measurements and decisions must
match before new arms execute on that checkpoint.

## Fixed algorithms

Every arm uses V18's exact split, V19's precomputed minibatches and four proposal
IDs per example, 200 interleaved plain SGD updates, learning rate .01, batch size
16 originals plus 16 second views, same 147-patch bank. Proposal selection is
maximum student cross-entropy; selected patches can differ as models diverge.
No access to evaluation labels, attacker roles, trigger identities or target
class in repair or admission. The historical evaluation trigger is contained
in the bank; this study cannot claim unseen-trigger generalization.

Let z0(x) be a frozen copy of the parent, z(x) the student, and xp the selected
patched version. K(a,b) = T^2 KL(softmax(a/T) || softmax(b/T)), T=2.

| Arm | Trainable parameters | Loss |
|---|---|---|
| hard_patch | All | Exact V19 joint clean/patch CE mean |
| anchored_clean | All linear layers; Conv2d weights/biases frozen | .5 CE(z(x),y) + 2 K(z0(x),z(x)) |
| anchored_patch | Same frozen convolutions | .5 CE(z(xp),y) + 2 K(z0(x),z(x)) + 2 K(z0(x),z(xp)) |
| anchored_suppress | Same frozen convolutions | anchored_patch + .1 mean_examples sum_(c != y) relu(z(xp)_c-z0(x)_c)^2 |

The pasted proposal did not define gamma or a complete anchored-clean objective.
We fix gamma=.1, sum non-label classes and average examples, and use clean CE in
anchored_clean. These are implementation choices, not tuned or validated values.
No suppression tuning follows failure. The fourth arm isolates suppression from
paired invariance; the first two do NOT isolate freezing from KL and changed CE.

The suppression term is the literal proposed raw-logit term. It is not invariant
to adding a class-independent offset to student logits. It can be nonzero on
clean models and does not identify a backdoor neuron. Its utility is empirical.
Distillation from a poisoned parent may preserve errors/backdoor behavior.
Frozen convolutions may retain a trigger representation. No early-layer safety
assumption is made. FEMNIST here has 62 total classes, not 62 other classes.

All arms present 6400 student training examples and 12800 proposal examples.
The anchored-clean arm observes patch proposals for matched accounting but its
loss uses clean examples only. Anchored arms add 3200 teacher examples. Frozen
layers change backward cost: equal example/update budgets are not equal FLOPs.
Record actual repair/evaluation wall time, parameter names, per-step component
losses, selected patches, parent/teacher/candidate/committed hashes. Student
transmissions remain 400 full models; additionally count one teacher copy per
repair client (5), with separate byte accounting. Network costs are simulated.

## Admission, evaluation and stopping

Use unmodified V18 private screen and verify panels, each five clients, disjoint
from the five repair clients. Paired clean-loss tolerance .02 + 2 SE, clean
accuracy tolerance 2 percentage points, nonincreasing worst-patch mean loss,
and noncollapse check all retained. Restore original panel iteration order to
preserve V19-fixed arithmetic. Both panels must pass; otherwise retain parent.
This is empirical validation, not a population guarantee. Repair providers are
assumed honest with correct labels. Historical private examples have already
been exposed in previous studies; no formal privacy guarantee is supplied.

Only after the verdict evaluate raw and committed global accuracy, non-target
trigger target rate, clean target rate, conditional ASR and its denominator,
lower-decile client utility and maximum prediction frequency. Report holds.
Global accuracy is an offline qualification target, never an admission input.

Primary candidate predeclared as anchored_suppress. Every seed must satisfy:
both clean and attacked models admitted; committed clean accuracy within 2pp of
clean source; attacked accuracy within 2pp of own source and matched clean source;
committed trigger target rate <=20%, improvement >=20pp from poisoned source;
at least 10pp improvement over raw anchored_clean and raw V19 hard_patch controls.
The proposal's below-10% value is separately reported as aspirational, not a
forecast. All other arms are reported even if the primary fails. No selection
of the best arm after inspecting attack outcomes. Complete all 24 diagnostic
cells unless integrity fails; no automatic tuning or full training afterwards.

A pass only motivates a separately specified end-to-end study: clean and gradual
attacks, repeated poisoning, honest exclusion and malicious accepted weight,
fresh exposure-audited data/seeds, unseen triggers, adaptive attackers and
fidelity-audited strong baselines. These checkpoint repairs do not measure any
of those effects. A fail rejects this fixed candidate, not all distillation.

## Proposal corrections and novelty

V19 demonstrates a modest ordering effect, not its asserted gradient mechanism.
One repair budget is not evidence of a plateau; CE gradients do not vanish merely
when the correct logit barely exceeds another. Data-size causal effects were not
isolated. Observed error differences must not be labeled causal percentages.
Admission 3/3 and trigger rate below 10% are unknown before measurement.

Known components include temperature distillation
https://arxiv.org/abs/1503.02531 and federated trigger-hardening defenses such as
FLIP https://arxiv.org/abs/2210.12873. Combining these ideas does not establish
novelty. The possible contribution concerns measured raw-versus-admitted repair
tradeoffs under heterogeneous private validation; novelty/generalization still
require comparison with CrowdGuard, DifFense and related private-validation and
backdoor-unlearning literature. No security guarantee is claimed.
