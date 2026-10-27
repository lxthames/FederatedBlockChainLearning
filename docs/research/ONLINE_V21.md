# V21: online feasibility of frozen V20 repair

Predeclared 21 October 2026, before any V21 trajectory outcomes. The user
authorized the online study following successful V20 checkpoint qualification.
Historical versions and stop rules are retained unchanged.

## Primary question

Does the fixed V20 repair preserve clean and gradual learning and control
repeated model-replacement poisoning when repair is deployed during training?
Endpoint checkpoint feasibility is established for the exposed V20 setting;
online efficacy and reinfection remain unknown. No coefficient or threshold
search is part of this experiment.

## Matrix and sequencing

36 cells: FEMNIST, development seeds 11/17/23, conditions none/low_and_slow/
model_replacement, arms clipping/current quorum/quorum plus V19 interleaved
hard-patch repair/quorum plus full V20 anchored_suppress repair. Every cell
runs 100 rounds from initialization under frozen V16/V17 training, attack,
partition, sampling, and quorum settings. Model replacement remains active
every round from its historical onset. New experiment IDs and V21 settings
identify this study; the inherited validated config retains its V17 protocol tag.

All 12 clean cells run first. Only after they finish, admit the 24 attacked cells
if each V20 clean seed has final accuracy within 2pp of the corresponding current
quorum, exactly 91 attempts and at least one accepted repair. The V19 comparator
is reported even if its clean utility fails. If the V20 gate fails, report
clean_rejected, hold the 24 attacked cells, and stop without revising the schedule.
This is one clean feasibility check, not an optimization over schedules.

## Fixed deployment rule

Use a short fixed warmup: no repair in rounds 1-9; attempt repair after every
normal aggregation in rounds 10-100 inclusive. Thus 91 attempts per repair cell.
This initial deployment choice is a heuristic, fixed before clean/attack runs;
it is not established as optimal or safe by V20. It receives a clean-only
qualification before attack execution. It does not read attack onset, ASR,
target label, roles, or evaluation outcomes. Every-round attempts allow the
experiment to measure immediate reinfection at substantial explicit overhead.

Keep the current per-update quorum and its actual normalized aggregation.
The pre-repair aggregate supplies the V20 frozen teacher. V19 uses exactly the
interleaved supervised hard-patch rule from V19-fixed. V20 uses unchanged T=2,
alpha=.5, clean KL weight=2, patch KL weight=2, raw-logit suppression weight=.1,
frozen Conv2d weights/biases, and trainable linear layers. Both use the same
precomputed per-round proposal/minibatch streams, 147-patch bank, 200 SGD steps,
learning rate .01, five repair clients and at most 32 examples per repair client.
Actual hard-patch selections may differ as models diverge.

Choose repair five/screen five/verify five noncontributors by the inherited
attack-agnostic allocation rule at each round. Within an attempt, clients and
examples are disjoint between all three groups. Across rounds, clients and
examples may recur and change roles. Examples used for repair can subsequently
appear in private update voting or validation. This repeated data use is a
deployment limitation and is not independent validation or formal privacy.
Honest correct-label repair providers and honest validators are assumed; their
corruption is outside this development study. All 30 noncontributors are
eligible for selection regardless of their simulated attacker role.
Selected clients provide correct-label repair and honest panel responses even
when their IDs belong to the simulated malicious local-training role. Thus
training-update attackers and repair-provider corruption are different threat
capabilities here; a deployment with malicious repair providers requires further
evaluation and cannot inherit this study's conclusions.

Apply the unchanged V18 paired clean-loss, 2pp private clean-accuracy,
nonincreasing worst-patch loss, and noncollapse checks on both panels.
Commit the scored repair only if both pass. Otherwise retain the pre-repair
quorum aggregate and record a repair hold. A repair hold does not roll back
the quorum's normal learning update or freeze the round. Global clean/attack
test scores do not enter repair, admission, schedule or checkpoint selection.
Global model parameters remain trainable for subsequent local federated training;
only the temporary repair student freezes its convolution parameters.

## Runner provenance and checks

The V21 runner is an explicit copy of the frozen V17 runner with a small
post-aggregation insertion and final-model exposure. build_runner.py reproduces
the complete file from declared replacements; tests enforce exact equality.
No historical file is modified. The aggregate is verified before repair;
the selected post-repair state is loaded exactly and verified before the ledger
commit, evaluation and next round. Its trace hash must match every selected
repair hash. Offline observers must preserve source/candidate/committed hashes.

Clipping and current-quorum controls must reproduce all 101 historical model
hashes and outcomes exactly. Online arms must match historical current-quorum
models for genesis through round 9 and match initial data, partition, participant
and local-training-work bindings throughout. Defense effects after round 10
naturally change subsequent updates and models; only their inputs/budgets remain
matched. Every final model has a serialized-state hash and completion seal.

Before launch: run meaningful unit/integration tests; reproduce both V19 and
V20 200-step clean-seed-11 round-100 checkpoint outcomes on the real GPU, including
their exact decisions. This smoke is a software check and not an online cell.
Every child process asserts transitive freezes. Source, policy, tests, launcher,
36 configs, historical controls and the V20 qualification/audit are frozen.

## Metrics and targets

Report every round's committed accuracy/ASR, accepted malicious aggregation
weight, update rejection counts and repair admissions/holds. After each repair
decision, independently evaluate source and candidate, recording raw and
committed metrics: global accuracy, non-target trigger target rate, conditional
ASR with its denominator, lower-decile client utility and prediction concentration.
No measurements are returned to the controller. Evaluate final models in all arms.

For targeted conditions, compute mean post-onset ASR over all 90 rounds 11-100,
including held repairs, and endpoint ASR. Record pre-repair trigger rate versus
previous round's committed rate to expose reinfection after one local-training
and aggregation step. End-of-round metrics do not imply continuous protection
during aggregation. Honest exclusion is computed from submitted clients and
accepted IDs, treating all clients as honest before attack onset/under none.
Accepted malicious sample weight describes the normal quorum aggregation;
repair changes the model and does not retroactively attribute its parameters.

V21 qualifies only if every seed has:

- Clean and gradual final accuracy each within 2pp of its current-quorum control.
- Targeted final accuracy within 2pp of the matched clean current-quorum source.
- Mean post-onset and endpoint ASR both <=20%.
- Improvement in each ASR measure >=20pp over current quorum.

Report V19 and clipping controls irrespective of their performance. Report all
per-seed raw/committed outcomes, lower-tail utility, conditional denominators,
prediction concentration and holds. Good mean ASR cannot rescue a failed seed.
Targets are empirical project gates, not formal guarantees or significance tests.

## Cost and interpretation

Count all inherited local training and quorum validation work, repair student
training/proposal/diagnostic presentations, teacher presentations, and both
models' private verification presentations. Each attempt simulates 400 student
model transfers; V20 adds five cached teacher deliveries, two models to every
panel member, one selected-model return and serialized private response bytes.
Response byte accounting is the actual JSON telemetry size and is an accounting
proxy, not an implemented privacy-preserving wire protocol. Report these terms
separately from total simulated transmitted bytes; no real distributed network
latency is measured. Full process wall time includes evaluations and file work;
round runtime includes additional offline evaluation; repair seconds are separate.
Example counts do not equate to FLOPs because V20 freezes convolution gradients.

Up to 18200 extra repair optimizer steps per repair cell, plus proposal and
validation work, are an important cost of this deployment choice. Private-data
reuse and the in-bank evaluation trigger limit claims even if development passes.
No unseen triggers, defense-aware attacker, fresh seeds or unseen evaluation data
are included. Freeze a separate independent confirmation study only after
development and novelty/fidelity assessment. No automatic threshold tuning,
schedule search, algorithm variants or confirmation launches follow V21.
