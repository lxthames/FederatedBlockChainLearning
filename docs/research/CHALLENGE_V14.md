# V14 perturbation and aggregate validation pilot

20 October 2026. Development pilot; no novelty, calibrated false-positive,
Byzantine-safety, or publication-readiness claim. Seeds 11/17/23 are exposed.
V13 is the immutable parent. New source: scripts/trust_challenge_v14; results:
results/challenge-v14; coordinator: artifacts/challenge-v14.

## Fixed scope

Maximum 60 runs: two datasets, three seeds, five arms, clean and the existing
targeted model-replacement attack. Four mechanism arms are current private
scaled-loss quorum, additional perturbation vote, aggregate gate, and both.
Clipping alone is an additional control. All use the V13 wide CNN, one local
epoch, SGD settings, data, sample weights, participants, fixed trust, clip ratio
two, and renormalized aggregation. Fashion: 30 rounds; FEMNIST: 40; onset: 11.
No monetary penalties. Honest committees only. No new attack variants in this
bounded pilot: unseen-pattern and defense-aware attacks are required follow-ups
if it passes. No automatic confirmation or full-experiment launch.

## Challenges and score

Each validator samples at most 16 of its PRIVATE validation examples, using an
independent deterministic random stream each round. The bank contains clean
images plus four transformations: randomly positioned black 2x2 patch, white
2x2 patch, gray 4x4 patch, and clipped Gaussian noise with standard deviation
0.08. Positions include all legal locations. The bank never reads the configured
attack location, target, attacker identities, evaluation data, or ASR. Some bank
patch sizes/colors overlap the existing attack family by design, so success is
not evidence of generalization to unseen trigger families. Randomness is
reproducible, not a cryptographic secret or adaptive-attacker protection.

For each perturbation and output class, measure the mean predicted probability
on transformed inputs minus clean inputs. Score a proposed model by the largest
positive increase in that quantity relative to the round's parent model. This
measures new perturbation-induced concentration, rather than absolute prediction
confidence. It is a heuristic; transformations may change a legitimate label.
No transformed-label loss is used by this additional score.

## Clean-only calibration, frozen before defense runs

First run the six clipping-only CLEAN controls with passive telemetry at both
individual nominal sample-share probes and the applied aggregate. Each control
must reproduce ALL V13 clipping clean model hashes; abort calibration otherwise.
By dataset and stage, freeze the higher empirical 99.5th percentile of those
clean validation scores. calibration.json binds all six completion digests and
thresholds; subsequent changes abort. No malicious examples or evaluation/ASR
labels choose thresholds. Pooling correlated rounds/votes supplies NO independent
false-positive guarantee. The same exposed seeds calibrate and screen; this is
development, not held-out calibration confirmation.

## Decisions

Candidate perturbation arm: require both the existing paired clean-loss vote
(mean increase <= two standard errors) and concentration score <= calibrated
candidate threshold, at the existing nominal sample-share probe. Existing
10-of-15 quorum decides candidate admission. Preserve the original clean-loss
vote separately in telemetry. The current arm has no additional challenge work.

Aggregate arm: after candidate admission and renormalization, apply the exact
proposed aggregate to a copy of the parent. A separate deterministic 15-member
panel evaluates its concentration score and paired clean loss on the same capped
clean examples. Require ten members to pass BOTH tests. If unsuccessful, commit
a zero update and no included client IDs for that round. Log a separate aggregate
veto; count the round as unsupported and all omitted updates as excluded. Do
not label this veto client misconduct or rewrite candidate committee decisions.
No group omission/repair in this first bounded pilot; it would add an additional
policy and substantial work. Combined arm uses both candidate and aggregate checks.

## Sequence and gates

Six clean calibration/control cells first, then all 24 clean mechanism cells,
then 30 possible attack cells. Each mechanism's attacks require ALL SIX of its
clean cells to pass: matched clipping control accuracy >=20%, <=5 percentage
point utility deficit against clipping, <=3 unsupported rounds, zero honest
monetary penalties. Attacks for failing mechanisms are held with reasons.
Clipping attacks always run. A held attack is not a successful defense result.

## Integrity and reporting

Source, V13 data/protocol bindings, this protocol and full manifest are frozen.
Each cell seals config, summary, split, events, trace, diagnostic-inputs and
challenge-settings, including calibration digest. All clean/attack comparisons
require identical models through round ten and matched training/data/participants.
Use all three matched seeds for means; show every seed and every failure/hold.
Report final ordinary accuracy, mean AND endpoint ASR, malicious acceptance,
honest exclusion, aggregate vetoes, full and added validation work and runtime.
Distinguish candidate admission from final inclusion after aggregate veto.
Do not interpret attack benefit unless attacks are effective against matched
clipping: mean post-onset ASR >= clean trigger-to-target rate +10 points.
Any advantage claim must also retain utility within five points of clipping and
show at least five points mean ASR reduction, with endpoint ASR visible.
The pilot's study report provides raw outcomes; benefit assessment is a separate
review after completion, not automatic promotion.

Full update archives remain disabled to bound storage. V14 records new behavioral
votes and applied-aggregate decisions, but individual-update causal replay still
requires separately archived tensors. These signals and the four-arm design test
an engineering hypothesis motivated by V13; literature overlap with CrowdGuard,
BackdoorIndicator, and FLIP must be resolved before any novelty claim.

## Reproduction

Run .venv-trust/Scripts/python.exe -m scripts.trust_challenge_v14.study --freeze,
then the same module with --run. Coordinator requires CUDA and preserves
interrupted result directories. No V13 source, protocol or results are edited.
