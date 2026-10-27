# V22: fixed periodic repair and private running-peak gate

Predeclared before V22 outcomes, 21 October 2026. This is a separate authorized
coordinator study; V21-fixed completes its 12 clean cells and holds its 24 attack
cells. No historical results, thresholds, stop rules or sources are overwritten.
CMT registration remains deferred by the user's earlier instruction.

## Question and controls

Test whether fewer repair invocations preserve learning, and whether a tighter
step check plus private running reference adds benefit. Neither success nor
novelty is assumed. The inner V20 loss, frozen convolutions, learning rate,
teacher definition and minibatch/proposal streams remain unchanged.

Matrix: FEMNIST seeds 11/17/23, none/low_and_slow/model_replacement, four arms:
clipping, current quorum, periodic V20 with original admission (v20_periodic),
and the same periodic V20 with the additional private gate (v20_peak).
There are 36 cells. The periodic-only comparator replaces V19 in this NEW matrix
to isolate the incremental admission change. V21 retains its original V19 arm.

## Fixed scheduling and inner repair

Attempt repair only after the aggregate in rounds 25, 50, 75 and 100. Exactly
four calls, 200 SGD updates each: 800 repair updates per repair cell. V21 used
18200 (91 x 200), not 36400. The 400 quantity is modeled student transmissions
per repair, not optimizer steps. Reducing calls saves 95.6% of repair invocations;
it does not imply a 95.6% reduction in total runtime, which includes federated
training, validation, monitoring, evaluation and file work.

Use unchanged full V20 anchored_suppress: T=2, alpha=.5, clean KL=2, patch KL=2,
suppression=.1, learning rate .01, five repair clients, 32-example cap, two passes,
20 steps per client, 16 originals plus 16 paired views, four hard proposals,
147 spatial patches. Original private screen/verify panels and all checks remain.
Current quorum still filters and aggregates updates; failed repair retains its
pre-repair aggregate. Normal federated training keeps all model layers trainable.

No 8% trigger prescreen is added: it lacks clean calibration and earlier V17
showed a prediction-flip blind spot. No schedule/threshold alternatives are
searched after observing V22 outcomes. Periodic repair may allow reinfection
between interventions; mean post-onset ASR includes all those rounds.

## Precisely defined private memory

The original proposal leaves EMA versus running peak and changing panels
ambiguous. This study chooses running peaks on FIXED examples for each validator.
It does not combine scores from different example sets as if they were paired.

For each of all 50 clients, choose at most 32 original private validation examples
once using the seed/client stream v22-fixed-private-examples. At every round,
score the PRE-REPAIR quorum aggregate on each fixed local set. Retain each
client's highest correct-count observation, including the current round.
These are observations along the defended trajectory, not an unobserved
quorum-only counterfactual. All clients supply honest monitoring reports even
if their IDs belong to the simulated malicious local-training role.

At each scheduled repair, use the original allocation of five repair, five
screen and five verify clients, disjoint within the attempt and excluding current
contributors. On each panel, obtain the candidate's accuracy on those panel
members' fixed monitor examples. Compute weighted pooled accuracies by total
sample count. Let A_source be current correct counts / n, A_candidate the paired
candidate value, and A_peak the sum of members' historical peak correct counts / n.

The ADDITIONAL check on EACH panel is:

    A_candidate >= max(A_source - .0025, A_peak - .015).

It is conjoined with every original V20 verifier check; it never overrides a
failure. The periodic-only arm computes the identical monitoring and proposed
gate telemetry but retains the original decision, giving the same information
and inference workload. Both arms have the same fixed schedule and inner budget.

A_peak combines per-client maxima that may occur at different rounds. It may
not be attainable by any single model, making the check conservative. With
roughly 100-160 examples per panel, .25pp is smaller than one correct prediction;
the effective step gate often permits no lost correct predictions. These are
fixed empirical choices, not calibrated false-positive guarantees. Subgroup
and global utility can still disagree with the private gate. No theorem that
this stops cumulative global harm is claimed.

Monitor examples can overlap earlier repair/voting examples, and clients can
change roles across rounds as in V21. This is adaptive reuse of private data,
not independent confirmation or formal privacy. No test set, attack identity,
target label or evaluation outcome enters monitoring or admission.

## Qualification and stopping

V21-fixed must finish with 12 completed clean cells, zero execution failures,
and clean_rejected before V22 starts. All 12 V22 clean cells precede any attacks.
The primary candidate is v20_peak. Every seed must finish within 2pp of current
quorum, execute exactly four attempts and admit at least one repair. An all-hold
candidate cannot qualify merely by reproducing the parent trajectory. If any
seed fails, hold all 24 NEW attack cells and stop. Do not loosen the gates.

If clean qualification passes, run all 24 attacked cells. Retain V21's final
targets: clean/gradual accuracy within 2pp of current quorum per seed; targeted
accuracy within 2pp of matched clean quorum; mean post-onset AND endpoint ASR
<=20%, each >=20pp better than current quorum. No endpoint-only success claim.
Report the periodic-only arm regardless of whether the primary succeeds.

The fixed V22 runner is explicitly derived from the V21 runner, adding only
private pre-repair monitoring. Controls must reproduce all 101 historical model
hashes. Online arms must match historical current quorum through round 24.
Same data, participants and training budgets throughout; after repair, models
and local gradients are expected to diverge. Preserve exact scored/committed
state checks, source/candidate immutability under offline evaluation, seals and
transitive freezes. Tests cover paired references, peak retention, missing rounds,
missing responses, step/cumulative vetoes, unchanged repair hashes, and actual
next-round commits. The previous real-GPU 400-step inner-loop smoke remains bound
and valid because no inner repair source is changed.

## Measurements and costs

All 100 round-level ASRs are recorded, including the intervals between repairs.
At each scheduled repair, evaluate source/candidate/committed models AFTER the
admission decision, with conditional ASR denominators, prediction concentration,
client utility and holds. Compare pre-repair ASR against the previous round's
committed ASR to expose reinfection. Report honest exclusions and accepted
malicious aggregation mass; neither is a certificate about repaired parameters.

In addition to V21's repair/quorum cost accounting, count every private reference
presentation and candidate fixed-example presentation. Count 50 additional model
deliveries per round conservatively, plus actual serialized reference responses,
and extra memory-gate responses. Original verifier model deliveries already cover
the candidate's private fixed-example scoring. Both periodic arms pay these costs.
No real network deployment, transport privacy or latency guarantee is provided.
Totals distinguish modeled communication, repair steps and wall-clock runtime.

## Interpretation of V21 and the pasted forecasts

V21 clean losses are 2.77, 3.96 and 4.20pp, with 50/51/53 admitted repairs.
But actual immediate global accuracy changes averaged +1.65 to +2.01pp per
accepted repair on those trajectories. Thus a small-negative-step ratchet is
not established by these data. Final loss divided by admissions is NOT the
per-intervention accuracy effect. Intervening local-training/aggregation changes
and repair changes can be separately accounted along the observed trajectory;
their causal mechanism remains unresolved. See the independent V21 trajectory
accounting artifact, rather than extrapolating a guaranteed four-call loss.

Four calls do not cap global damage at .22-.31pp and do not inherit V20's 9.23%
endpoint trigger rate. The tighter gate may reject useful repairs or every
repair. This study tests those possibilities. A positive development result
would still need novelty/fidelity audits, unseen triggers, defense-aware attacks,
fresh exposure-audited data/seeds and independent confirmation. No acceptance
readiness or guaranteed deadline feasibility follows from this implementation.
