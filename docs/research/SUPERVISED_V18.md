# V18: bounded supervised checkpoint repair

This is a new, separately frozen feasibility experiment following the rejected V17 diagnostic. V16/V17 sources, thresholds, completion seals, qualification decisions and results remain unchanged. V18 does not relax the V17 gate or automatically launch an end-to-end defense study.

## Question and scope

Can correct-label patch repair reduce the backdoor in a fixed current-quorum model while retaining useful classification, beyond equal-budget clean fine-tuning and random augmentation? This is a hypothesis. Filtering, hardening, adversarial augmentation and trigger repair are established ideas; this experiment establishes neither novelty nor acceptance readiness. FLIP (ICLR 2023, https://arxiv.org/abs/2210.12873) is a required related-work comparison before any new-defense claim.

Use FEMNIST development seeds 11/17/23, clean and model-replacement conditions. Reproduce all six unchanged current-quorum trajectories for 100 rounds. Capture rounds 11, 30 and 100, fixed before new repair outcomes. Round 11 examines observed early class collapse; rounds 30 and 100 are the mature-checkpoint feasibility endpoints. Match every initial/round model hash, data/splits, participants and training work to the frozen V16-fixed references before using a checkpoint. Replay preserves existing training/attack schedules.

There are 24 subprocess jobs: six replays followed by 18 checkpoint experiments, clean checkpoint experiments first. Each checkpoint scores four arms (72 results): unchanged, clean_tune, random_patch, hard_patch. The three trained arms use identical optimizer steps, base examples, proposal count, learning rate and model transfer budgets. No parameter search or selection by attack outcomes occurs.

## Data and trust assumptions

At each checkpoint select 15 noncontributor clients without replacement, using a new independent seed/checkpoint stream: five repair clients, five screening clients and five verification clients. Neither client selection nor example selection reads attack identities or target labels. For each selected client, deterministically select up to 32 of its original private validation examples (at least two required). Client sets and example indices are disjoint across all three roles. Freeze and hash all index assignments.

The repair role now trains on its selected private validation examples. This is additional labelled-data use and additional computation compared with the current quorum. The repair clients are assumed to execute correct-label repair faithfully; verification clients are honest, matching the historical zero-malicious-validator setting. This stronger repair capability is explicit, not an adversarial-validator guarantee. No poisoning identities are used to find honest repair clients. The simulator supplies clean original validation labels; evaluating poisoned repair providers is future work.

Verification examples never enter repair or hyperparameter selection. They were previously used in historical quorum voting, so disjointness from this repair does not make them fresh confirmation data. The official development evaluation arrays remain evaluation-only; neither their features/labels nor ASR enter repair or admission. Seeds, trigger family and development test data are already exposed. No privacy guarantee follows from executing model training locally: models/updates can leak data.

## Repair rule

Sequential local model fine-tuning on the five repair clients, in client-ID order, two passes, 20 plain-SGD steps/client/pass, learning rate 0.01, no momentum, batch size 16. Thus each tuned arm has 200 optimizer steps and 6,400 training example presentations. Each step uses the same 16 clean base examples and a second 16-example view. Labels always retain their original correct values.

Reuse the fixed 147 spatial patches from V16/V17, excluding Gaussian noise. Patch labels are assumed invariant; small patches can erase informative handwriting strokes, and any resulting utility loss must be reported. This bank includes the existing evaluation trigger, so this is a known-family feasibility experiment, not unseen-trigger evidence.

Four proposals per example/step, sampled uniformly from the fixed bank with replacement. The hard_patch arm chooses the proposal with maximum supervised cross-entropy for each example. The random_patch arm uses the first sampled proposal. The clean_tune control evaluates four duplicate clean proposals and trains on two copies of the clean batch. All three arms pay the same number of forward examples and gradient examples. Report runtime separately; control duplication is an experimental equal-work control and should not be advertised as an optimized baseline. All proposal decisions use correct repair labels and model predictions, without any attacker target or test data.

GPU forward/backward computation remains deterministic. Plain-SGD parameter subtraction is computed independently on CPU from the gradients and loaded with exact state equality checks. Parent models remain unchanged. Restore CPU/CUDA RNG streams after each repair. Save every repair step's loss, selected transformation IDs and resulting model hash. Sequential model exchange is simulated; network transfer is accounted for, not physically deployed.

## Verification and commitment

All 147 patches are evaluated on every screening/verification example. Record supervised clean loss, maximum patch loss per example, clean accuracy, worst-patch accuracy, class counts and exact model hashes. Each panel pools its five clients' example losses. It passes only when:

1. Paired mean clean-loss increase is at most 0.02 plus two empirical standard errors.
2. Clean accuracy drops by at most two percentage points.
3. Mean per-example worst-patch loss does not increase (numerical tolerance 1e-12).
4. Predictions do not collapse to a single class when the pooled true labels contain multiple classes.

Both panels must pass. Pooling and the SE allowance are empirical rules with no false-positive/security guarantee or population coverage guarantee. The collapse check cannot certify general model utility. A repaired candidate is committed only with exact equality to the scored hash; otherwise the parent is retained and a hold is recorded. The unchanged arm is a reference trajectory, not a gated deployment intervention. Its diagnostic verification can fail while its unchanged model is still reported as the reference.

Report raw candidate outcomes and committed outcomes for every arm, including rejected repairs. Offline evaluation follows the admission decision and reports task accuracy, trigger target rate, clean-input target rate, ASR conditional on correct clean classification, its denominator, prediction concentration and lower-decile client utility. Clean-condition trigger target rates are controls, not attack successes. Evaluation uses the historical trigger/target only in the offline evaluation function. Report all model/data transfers, repair work, verification work, elapsed time and artifact sizes. Runtime includes simulation/hashing overhead and is not a deployment latency measurement.

## Fixed advancement rule

All 72 results must exist with valid seals and equal trained-arm budgets. For hard_patch, at both mature checkpoints (30 and 100), on each seed:

- Clean-condition committed accuracy is within two percentage points below its unchanged reference, and both panels admit it.
- Attacked-condition committed accuracy is within two points below both the attacked unchanged reference and the matched clean unchanged reference, and both panels admit it.
- Attacked committed trigger target rate is at most 20%, with at least 20 percentage points reduction from the attacked unchanged reference.
- It has at least 10 percentage points lower trigger target rate than each RAW equal-budget clean_tune/random_patch control. Using raw controls prevents the admission gate from manufacturing an apparent repair advantage by rejecting a good control.

These are checkpoint feasibility targets, not mean post-onset or endpoint ASR for a retrained defense. Round 11 results and all failures are reported; excluding round 11 from this mature feasibility gate does not resolve deployment during early training. A passing report is named checkpoint_promising, and a failing report checkpoint_rejected. Stop after this experiment; do not tune thresholds or launch another variant automatically. Even success still requires clean/gradual/continuous-poisoning end-to-end comparisons, early-training behavior, strong baselines, overhead, an adaptive attacker and independently held-out confirmation before a new-defense claim.

## Reproducibility

Freeze V18 code, this policy, tests, launcher, historical freeze, reference completion/trace/summary hashes, the data bindings and complete job manifest. All jobs run in isolated subprocesses with coordinator/cell OS locks. Match all 606 replay model hashes. Each completed job seals every output file. Preserve interrupted output directories inside the V18 root. Refuse altered inputs, checkpoints, seals or settings. Status and reports use atomic JSON replacement. No historical artifact is overwritten.
