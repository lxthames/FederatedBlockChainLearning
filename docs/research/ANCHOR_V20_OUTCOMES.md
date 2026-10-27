# V20 outcome and claim decision

Completed 21 October 2026, 20:57:42 Taipei. Status: checkpoint_promising.
All 24 cells completed with zero execution failures. This is the result of the
predeclared full candidate, not post-hoc selection of the strongest ablation.

Sources: [sealed comparison](../../artifacts/anchor-v20/comparison.md),
[independent completion audit](../../artifacts/anchor-v20/audit.md),
[per-seed qualification](../../artifacts/anchor-v20/analysis.json).

| Method | Clean raw accuracy | Clean admits | Poisoned raw trigger rate | Poisoned admits | Committed trigger rate |
|---|---:|---:|---:|---:|---:|
| V19 interleaved hard patch | 74.92% | 1/3 | 50.50% | 2/3 | 69.21% |
| Frozen-convolution clean anchoring | 78.82% | 2/3 | 99.96% | 1/3 | 99.96% |
| Paired patch invariance | 78.08% | 3/3 | 15.79% | 2/3 | 45.69% |
| Full V20 with suppression | 77.59% | 3/3 | 9.23% | 3/3 | 9.23% |

Clean source accuracy is 78.54%. Full V20 clean accuracy loss is .95pp on average,
with per-seed losses .76, 1.05 and 1.05pp. Poisoned-source accuracy is 77.46%;
full V20 reaches 77.41%. Every seed passes the declared utility limits, including
comparison of the repaired attacked model with its matched clean source.

Full V20 committed trigger target rates by seed: 11 = 10.31%, 17 = 7.78%,
23 = 9.61%. Each is below the frozen 20% gate and exceeds the required
improvements over the poisoned source, raw clean anchoring, and raw V19 controls.
The average is below 10%; not every seed is. Conditional ASR on correctly
classified clean non-target examples is 7.45%, 3.96%, 6.22% respectively, with
denominators 1154, 1162, 1286. Do not interchange those denominators with the
reported all-non-target trigger rate.

This reduction is not explained by collapse to a constant clean prediction:
attacked clean accuracy remains 75.74-79.68%, maximum prediction frequency
6.74-8.35%, and lower-decile client utility 59.09-64.09%. These metrics do not
guarantee that every subgroup is unharmed; retain all client-level results.

The paired-invariance ablation still rejects attacked seed 17, keeping its
committed trigger rate at 99.94% despite a raw rate of 10.24%. The suppression
variant improves raw removal and permits admission in this seed. Therefore the
full improvement is not solely an effect of admitting more repairs. Freezing
and clean anchoring remain bundled, so their individual causes are unresolved.

## Integrity and cost

All six baseline cells reproduce V19 exactly, covering 1200 optimizer updates.
All 48 candidate/committed saved states match their hashes and score records;
all 72 checked frozen convolution tensors match the source checkpoints.
Completion seals and transitive source/data freezes match. Relevant test suite:
87 passed; the 10 new tests passed again after removing a test-only warning.

Student budgets match at 200 optimizer steps, 6400 training presentations and
12800 proposal presentations. Anchored methods add 3200 teacher presentations
and five modeled teacher transfers per cell, atop 400 student transfers. Frozen
convolutions reduce gradient computation, so equal presentations do not mean
equal FLOPs. Mean measured cell time (repair, verification and evaluation) is
approximately 3.62s for V19 and 3.59s for full V20 on this machine. These are
single-run timings, not evidence of a speedup or real-network latency.

## Defensible present claim

On six exposed FEMNIST round-100 checkpoints, the fixed frozen-convolution,
teacher-anchored patch repair with logit suppression reduces average committed
non-target trigger rate from 69.21% for V19 to 9.23%, with all private panels
admitting the full candidate and per-seed utility within the declared limits.

This is evidence of a promising checkpoint repair algorithm. It is not yet
evidence of a publishable online federated defense. No gradual-attack result,
continual-poisoning resilience, unseen-trigger result or independent confirmation
has been measured for V20. The evaluation trigger lies in the known patch bank.
The teacher and convolution assumptions, raw-logit shift sensitivity, honest
repair providers, and exposed development data remain limitations.

## Next evidence, separately frozen

1. Freeze the full candidate and its deployment rule before online evaluation:
   when repair is invoked, which model is the frozen teacher, and what happens
   after a veto must be explicit. A successful endpoint repair does not answer
   these questions or establish a successful per-round policy.
2. Use matched clipping/current-quorum/repair controls in clean, gradual and
   repeated targeted conditions. Measure utility, post-onset and endpoint attack
   rates, holds, honest exclusion, accepted malicious weight and overhead.
3. Complete the novelty and fidelity audits, including an applicable
   distillation-repair comparison as well as FedVal, FLAME and CrowdGuard.
4. Independently confirm with exposure-audited data/seeds, held-out trigger
   families and an attacker aware of the frozen repair/verification rule.

No new experiments were automatically launched after this diagnostic. Its
predeclared stop point has been reached successfully. No claim that acceptance
requires only minor changes is justified.
