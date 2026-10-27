# V19-fixed: schedule-by-objective diagnostic

## Fixed scope

Test whether scheduling the existing repair updates differently changes utility and backdoor outcomes. This is a causal comparison of a scheduling intervention on the exposed FEMNIST development checkpoints. It does not establish catastrophic forgetting, stroke occlusion, a novel defense or protection under continuing poisoning by itself.

Use V18's exact round-100 parent models, client allocations and examples for seeds 11/17/23, in the clean and model-replacement conditions. Historical sources/results and V18's rejected qualification remain unchanged. No replay of the 100-round trajectories is needed: use their already captured, sealed states after verifying their bindings.

The 2x2 factors are schedule (blocked sequential versus round-robin interleaving) and objective (clean_tune versus hard_patch). Across three seeds and two conditions this gives 24 factorial arm results. Twelve subprocess jobs each run the two objectives. First run all six sequential jobs and require exact V18 reproduction for every optimizer step, selected transformation, loss, raw/committed evaluation, verdict and historical work count. Only then permit the six interleaved jobs. All sequential/interleaved jobs are clean-first within schedule. No result-dependent configuration selection occurs.

## Treatment and fixed inputs

Generate each client's complete minibatch and proposal stream before running either treatment, using V18's original domain, seed/checkpoint, epoch/client keys, first RNG-seed draw, sample selections and four proposal draws. Save the common plan, its SHA256, client IDs, within-client sample indices and proposal IDs. The canonical plan is ordered (epoch, client, step). It must be identical across schedules and objectives for a matched checkpoint. Independent hard-patch selection can change because the current model changes; log every selected transformation as a treatment response, not a changed input bank.

Sequential schedule: for each of two passes, visit five clients in sorted ID order and execute 20 updates/client. Interleaved schedule: for each pass and each of 20 local-step indices, visit the same five clients in sorted ID order and perform that client's next update. Both perform 200 parameter updates. Neither averages five gradients into a single update nor changes the learning rate.

Inherit V18's five repair clients and two disjoint five-client verification panels, at most 32 examples/client, batch size 16, four patch proposals, the same 147 spatial transformations, learning rate 0.01 and plain SGD with no momentum. Each update trains on 16 clean examples and 16 second-view examples with unchanged correct labels. Clean tuning's second view is clean; hard_patch selects the largest supervised-loss proposal per example. Both objectives have identical training and proposal-inference budgets. Weight arithmetic is computed on CPU from GPU gradients and loaded with exact equality checks. Reject architectures with dropout or batch-normalization state; the frozen wide CNN has neither. Restore RNG state after repair.

## Passive observations and overhead

At update 0 and every 20 updates, evaluate clean loss and accuracy on every repair client's selected examples. These measurements never use verification or test data and never select an intermediate model or stop training. Both schedules incur the same diagnostic example count. Every optimizer step logs owner, loss, chosen transformation IDs and model hash. These observations can expose client-local changes, but are training-side measurements and cannot certify generalization or prove a forgetting mechanism alone.

Count model exchange whenever the active client/pass owner changes: two full model transmissions per client visit. Sequential repair has 20 transmissions; interleaved repair has 400. Thus update/example budgets match while modeled communication differs by 20x. Model transfers are simulated rather than physically performed; measured runtime includes Python, integrity checks and passive diagnostics and does not estimate distributed deployment latency. Report transfer counts and bytes, not an equal-network-overhead claim.

## Decisions and evaluation

Use the original V18 verification function unchanged: two panels must both pass the paired clean-loss allowance (0.02 plus two empirical SEs), no more than two percentage points of private clean accuracy loss, nonincreasing worst-patch loss, and rejection of single-class predictions when pooled true labels contain several classes. Verification is empirical, with no formal population/false-positive guarantee. Verify the model hash before committing the repaired state; otherwise retain the parent and record a hold. No test metric or attacker identity enters repair or admission.

Measure raw and committed task accuracy, trigger target rate, clean target rate, conditional ASR and denominator, lower-decile utility and client utilities after admission. Include every failure. In clean runs the trigger target rate is a control rather than attack success. Classify all data as exposed development data; these are not fresh seeds, unseen triggers or independent confirmation. Repair training assumes faithful execution on original correctly labelled private data and inherits V18's additional data-access/trust assumptions. No formal privacy guarantee applies.

For each seed/condition/objective, compute the paired interleaved-minus-sequential raw and committed accuracy differences and sequential-minus-interleaved trigger-rate differences. Compute schedule-by-objective interactions by subtracting the clean-tuning schedule effect from the hard-patch schedule effect. Report all per-seed effects. Do not treat three seeds as a population-level causal theorem or equate an endpoint improvement with proven catastrophic forgetting. No prescribed direction of effect is assumed.

## Feasibility targets and stopping

The proposed arm is interleaved hard_patch. At round 100, every seed must have both clean and attacked repairs admitted; committed utility must be within two percentage points below the unchanged clean reference, and attacked utility must also be within two points below its attacked reference. Attacked committed ASR must be <=20%, with >=20pp improvement over the unchanged attacked model. Preserve V18's >=10pp advantage over raw equal-work clean tuning and random-patch controls: use interleaved clean tuning and the sealed V18 random-patch result. The random-patch comparison is cross-schedule and cannot isolate a scheduling effect on random augmentation.

If all targets pass, report checkpoint_promising; otherwise completed_targets_not_met. Separately report measured scheduling effects even if these targets fail. Passing applies only to this round-100 feasibility test and does not reverse V18's all-checkpoint qualification or establish a publishable defense. Stop after the fixed matrix. No automatic projection/KL study, threshold search, new variant, continued-poisoning study or confirmation run follows. Such work requires a separate frozen protocol.

## Integrity

Freeze V19 sources, tests, launcher, this policy, complete manifest, inherited policy, V18 freeze, and all six source experiment plus six source replay completion hashes. Verify complete source seals before using any data or model state. Match the exact V18 partition bindings. Use CPU model hash/state checks, scored/committed equality, isolated subprocesses, OS coordinator/cell locks and atomic status/report JSON. Seal every completed output file. Resume only matching sealed jobs; preserve interrupted jobs inside the V19 output root. Refuse changed frozen inputs or failed baseline parity. Historical artifacts remain immutable.

## Ordering correction

The initial V19 launch failed an exact baseline audit before any interleaved result was generated. Restore the original allocated client order after loading sorted JSON measurements, preserving the V18 floating-point reduction order. No mathematical rule, threshold, patch, training step or outcome-selection policy changes. Preserve initial V19 source/freeze/output and bind its freeze plus INTERLEAVE_V19_ORDER_AUDIT.md in this new freeze. The fixed module imports the unchanged frozen V19 repair engine. All output uses separate interleave-v19-fixed roots.
