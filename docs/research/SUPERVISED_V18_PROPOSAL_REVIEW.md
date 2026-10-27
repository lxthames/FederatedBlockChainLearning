# Review of the proposed verification/repair explanation

Date: 2026-10-21. This is a post-hoc interpretation and proposed diagnostic, not a change to the frozen V18 protocol or qualification. No new training results are asserted.

## What the existing controls establish

Round-100 FEMNIST, means across development seeds 11/17/23, raw models before admission:

| Repair | Clean-model accuracy change | Attacked-model accuracy change | Attacked-model ASR |
|---|---:|---:|---:|
| Clean fine-tuning | -3.541 pp | -1.233 pp | 99.956% |
| Random patches | -3.596 pp | -0.952 pp | 96.101% |
| Difficult patches | -4.395 pp | -1.581 pp | 47.448% |

Difficult patches add 0.854 pp of clean-model accuracy loss beyond clean fine-tuning and 0.348 pp on attacked models. The clean-versus-attacked asymmetry already exists with no patched training. Therefore the proposed account that stroke occlusion explains the entire 4.40 pp loss is not established. The incremental comparison does not quantify independent causal shares; scheduling, data distribution, objective and initialization may interact.

The difficult-patch repair does have a substantial effect beyond both equal-work controls on the exposed round-100 checkpoints. This is a checkpoint result, not protection under continuing poisoning.

Private panels reject seeds 17/23 because screening-panel accuracy changes are -2.50/-3.185 pp. Their global evaluation accuracy changes are -1.361/-1.868 pp. These are different populations. The disagreement does not prove that either panel is wrong, and real subgroup harm may be hidden by the global average. Seed 11 is admitted; actual committed mean ASR is 84.010%. On the clean seed-23 model, an admitted difficult-patch repair loses 3.853 pp of global accuracy, showing that panel approval does not guarantee the evaluation utility bound either.

## Claims requiring correction

- "Verification-repair deadlock" can be an informal description of observed holds. A dynamical deadlock, feature-space conflict or novel general mechanism has not been demonstrated by a one-shot checkpoint test.
- The inbound quorum uses scaled, paired clean cross-entropy loss, not simply clean accuracy. The outgoing verifier uses clean accuracy, paired clean loss, worst-patch loss and a constant-prediction check.
- Shortcut neurons, orthogonal clean/backdoor subspaces, selective early gradient absorption and reconstruction of an earlier backdoor by later clients have not been measured. They are hypotheses.
- An endpoint of 47.45% is not evidence of a plateau. A training trajectory or convergence analysis is needed.
- The fixed 147-patch bank contains the existing FEMNIST trigger. Poor sampling/optimization or incomplete coverage of examples/classes may matter, but absence of the true trigger from the bank is not the explanation.
- The fraction two rejected out of three applies to these particular round-100 difficult-patch candidates. It is not a population estimate for standard repairs.
- A0 plus three single-factor changes is a one-factor-at-a-time design, not a 2x2 factorial. Three binary factors need eight combinations to estimate their interactions fully.
- Synchronous gradient averaging and interleaving are different interventions. Averaging changes update frequency and effective step scale unless separately controlled. Oversampling changes the empirical objective and cannot create missing classes or additional independent examples.
- Restricting patches to low-salience regions changes challenge coverage. Any utility benefit must be assessed alongside the attacks that become uncovered.

## Projection: precise local statement

Let the update be theta_new = theta - eta*d, and let g_c be the gradient of a chosen clean-batch loss at the current theta. Then its first-order change is -eta * dot(g_c, d).

For nonzero g_c, exact orthogonal projection of a repair gradient onto the normal plane of g_c gives dot(g_c,d)=0. This protects that scalar batch loss only to first order. It does not identify the space of all clean features, preserve classification accuracy, control finite-step curvature, or protect unrepresented clients. Near-zero clean gradients make this constraint uninformative. Adding epsilon in the denominator also means exact orthogonality no longer holds in general.

A less restrictive local condition is dot(g_c,d)>=0, retaining gradient components that already decrease clean loss and projecting only conflicting components. With several repair-client clean gradients, simultaneous half-space constraints can be studied; satisfying one client's condition need not satisfy the others. Clean gradients must come from repair-side data, never from the disjoint verification panels. Such methods require clean-only parameter selection, numerical checks and subsequent empirical utility verification.

Gradient projection is established in PCGrad and GEM; combining it with repair is not sufficient novelty:

- PCGrad / Gradient Surgery for Multi-Task Learning: https://arxiv.org/abs/2001.06782
- Gradient Episodic Memory: https://arxiv.org/abs/1706.08840

## Other proposed remedies

Clean-input KL anchoring is a plausible regularizer. It constrains sampled predictions rather than unseen classes, and a poisoned teacher can preserve undesirable behavior. It cannot be described as preventing forgetting without measurement. Retain correct-label supervision and test KL alone, projection alone and their combination if pursuing that hypothesis.

Layer freezing should be an ablation, not an assumed localization of the backdoor. No layer attribution has been established in this experiment. A near-zero patch-flip score must not authorize bypassing protection: V17 already contains a counterexample where clean and patched inputs both collapse to the target, giving zero contrast.

Client hardening and trigger-based repair also overlap FLIP: https://arxiv.org/abs/2210.12873. Priority and fidelity audits remain necessary.

## Recommended next bounded causal comparison

First examine the utility damage already present without patched training. Use a genuine 2x2 design at the fixed round-100 checkpoints:

| Schedule | Clean-only objective | Difficult-patch objective |
|---|---|---|
| V18 blocked sequential schedule | existing control | existing candidate |
| Round-robin interleaved schedule | new matched control | new matched candidate |

For each seed and clean/attacked condition, use the same parent hash, clients, examples, label multiset, number of updates, learning rate and precomputed per-client batches and patch proposal IDs. Interleave the same 200 per-client minibatch updates, rather than changing the learning rate or replacing them with 40 averaged updates. Model-dependent hard-patch selections may differ; log them as part of the treatment response. Verify the baseline implementation reproduces V18 before interpreting the new schedule. This comparison has 24 arm/condition/seed cells, of which 12 baseline cells already exist.

Record per-client changes on repair-side clean examples after each client block, gradient alignment and final independent-panel outcomes. Any test/ASR trajectory is evaluated offline after the fixed run and cannot select an intermediate model. Report matched per-seed differences, every hold and both raw/committed outcomes. Use the existing utility and ASR requirements; do not retrospectively relax V18 or assert a mechanism from success on one seed.

This isolates a scheduling intervention and its interaction with the patch objective. It does not by itself isolate scarcity, missing classes or stroke occlusion. A finding about the dominant cause requires additional controlled evidence. Projection/KL should be separately ablated if pursued, with equal compute/data access accounted for.

## Paper language

Supported observation: in this exposed FEMNIST checkpoint study, difficult-patch repair reduces backdoor success beyond equal-work controls, but clean-data degradation and disagreement between private-panel and global evaluation utility limit committed protection.

Unsupported conclusions: a newly discovered deadlock, a proven clean/backdoor subspace decomposition, guaranteed three-of-three admission, guaranteed ASR below 20%, or acceptance readiness.

The official CFP includes evaluation frameworks and backdoor defense, with abstract registration October 24 and paper submission October 31, 2026. Full papers use ACM sigconf, at most nine pages including appendices, plus references and the GenAI Usage Disclosure section. Venue scope does not establish contribution strength: https://eigtrust.acm.org/trust2027/cfp/
