# V23: separately frozen periodic-only attack evaluation

Predeclared 22 October 2026 (Asia/Taipei), before new V23 attack outcomes.
The user authorized the six-trajectory study after reviewing V22's clean results.
V22's primary candidate remains failed: 12 clean cells completed and 24 attacks
held. V23 does not change V22's gate, status, files or blocked manifest.

## Selection and evidence scope

Promote V22's v20_periodic comparison arm as the V23 primary candidate after
observing its exposed development clean results: accuracy deltas -.2520,
-.3094, -.2919pp versus current quorum; admissions 3/4, 4/4, 4/4. This is an
explicit clean-outcome-informed selection, not independent confirmation or an
originally successful V22 primary. No V22 attack trajectories were run.

Six new 100-round FEMNIST trajectories from initialization: three seeds
11/17/23 under repeated model_replacement, then three under low_and_slow.
All six finish irrespective of performance failures; only software/integrity
failures stop the coordinator. No threshold, schedule or loss tuning follows.

Reuse 21 comparison rows: nine sealed V22 clean results (clipping/current quorum/
periodic-only V20 for each seed) and twelve sealed V16-fixed attack controls
(clipping/current quorum x two attacks x three seeds). Reports comprise 27 rows
but only SIX are new trajectories. Reused rows are never counted as fresh or
independent evidence. Existing exact V21/V22 reproduction checks, historical
seals and per-trajectory configuration/matching checks establish reuse validity.

## Frozen candidate

Use the existing V22 runner and v20_periodic controller without source edits.
Repair after normal quorum aggregation only in rounds 25, 50, 75, 100. Each call
uses unchanged full V20: frozen convolutional repair student, paired clean/patch
KL teacher anchoring, raw-logit suppression, 200 SGD updates and original private
screen/verify checks. Teacher is the pre-repair aggregate. Both panels must pass;
otherwise retain that aggregate. No memory gate, omission search or attack-aware
triggering is applied. All model layers remain trainable for subsequent ordinary
federated training. The target and poisoning schedule are unchanged from controls.

Retain V22's passive fixed-example monitoring and hypothetical memory-gate
measurements, although they cannot change admission for this arm. This preserves
the exact algorithm/workload of the reused clean evidence. Count their complete
cost. Removing that overhead would be a new implementation requiring separate
matching checks; V23 makes no optimized-overhead claim.

Inherited provider assumptions remain: correct-label honest repair, honest panel
and monitoring reports even for IDs with a malicious local-training role, no
formal privacy guarantee. Repair/panel clients and examples are disjoint within
an attempt; across rounds they can be reused and change roles. The known
evaluation trigger lies in the patch bank. Evaluation examples and target labels
remain unavailable to repair, scheduling and admission.

## Reuse and integrity audit

Before freezing and launching, verify every reused completion seal and identity.
For V16 controls verify the exact original configuration and 100-round integrity
summary. Reconstruct missing accepted malicious mass from historical local
sample counts and accepted IDs, using the V21-fixed audited helper. No missing
historical conditional-ASR metric is fabricated.

Each new trajectory must match its current-quorum reference configuration except
experiment ID/protocol label, initial model/data/test/partition/participation and
local training budget. It must reproduce all 25 historical model hashes from
genesis through round 24, including poisoning before the first repair. Later
models and gradients naturally diverge as repair takes effect. Verify all 100
global commits, all four scored/selected repair hashes and the final serialized
state. Verify the periodic-only decision equals the original verifier decision;
the hypothetical memory gate must never be applied. Freeze transitive V22/V20
sources, new sources/tests/policy/launcher, six configs and all reference seals.

## Qualification, measured on every seed

The reused periodic-only clean evidence must first satisfy within-2pp final
accuracy versus current quorum, four repair attempts and at least one admission.
The V23 development gate then requires all of:

- Clean final accuracy within 2pp of current quorum.
- Gradual-attack final accuracy within 2pp of current quorum under that attack.
- Targeted final accuracy within 2pp of the matched CLEAN current-quorum result.
- Targeted mean post-onset ASR and endpoint ASR both <=20%.
- Each ASR measure improves by >=20pp versus current quorum.

Mean post-onset ASR includes ALL 90 rounds 11-100, including intervals between
repairs and vetoes. A successful endpoint or raw candidate cannot substitute for
that mean or for the actual committed model. Report clipping alongside current
quorum, every seed, all holds and utility costs. No averaging rescues a failed
seed. Passing this exposed development gate does not establish publication
readiness or transfer the previous V20 checkpoint result to unseen attacks.

## Reporting

Retain full committed accuracy/ASR trajectories. After each final verdict,
independently evaluate source/candidate/committed models for clean utility,
non-target trigger target rate, conditional ASR and its denominator, prediction
concentration and lower-decile client utility. Final candidate metrics are
evaluated independently. Historical controls retain only their actual recorded
metric coverage, clearly labeled reused.

Record source ASR versus the previous round's committed ASR at each scheduled
repair, as well as ordinary round-level ASR throughout the intervals. Report
honest-update exclusion, accepted malicious aggregation weight, private decisions,
repair holds, local training, proposal/teacher/verifier/reference presentations,
modeled model/response bytes and measured runtime. Extra passive monitoring
costs are included. Simulated bytes are not real network latency measurements.

## Stop and publication boundary

Finish the six trajectories unless integrity fails. On a failed empirical gate,
record completed_targets_not_met and stop this fixed candidate without automatic
variants or tuned schedules. A pass only motivates a separately frozen independent
study with fresh exposure-audited data/seeds, held-out triggers, a defense-aware
attacker, a novelty audit and fidelity-checked strong comparisons. The three
seeds and development evaluation data are already exposed. No new abstract
registration or external submission is part of this authorization.
