# Initial V19 baseline-order audit

The initial V19 coordinator stopped on the first sequential clean-tuning baseline before generating any interleaved result. Keep its code, frozen-inputs.json, status and incomplete outputs intact. This is a serialization/reduction-order discrepancy, not an observed scheduling treatment effect.

The 200 optimizer losses, selected transformations, per-step model hashes, final candidate, raw evaluation and committed evaluation matched V18 exactly. The strict verdict-dictionary equality check failed on one floating-point standard error. V18 computed the screening clean-loss standard error as 0.08161944178953527; the reloaded measurement order produced 0.08161944178953524. Both rejected the repair under the same rule.

Cause: V18's original screening membership order is [40, 5, 30, 47, 15], and its verification order is [27, 23, 35, 8, 19] for clean seed 11 at round 100. Saving measurement dictionaries with sorted JSON keys and reloading them changed the reduction order. The paired-loss function iterates the parent-measurement dictionary, so bit-exact arithmetic summaries require restoring the allocated order. Merely matching the unordered client set is insufficient for exact provenance reproduction.

Offline audit using the sealed V18 measurement arrays confirmed that restoring the allocated order reproduces the complete historical verdict exactly. No model was selected, trained differently or admitted because of this correction.

Create a separate trust_interleave_v19_fixed module, artifact/result roots, tests, policy and launcher. Reuse the original frozen scheduling repair engine. Restore both saved parent panels to the original allocate() client order and reject mismatched member sets. Keep all source experiment conditions, thresholds, datasets, example order, training budgets and inference rules unchanged. Bind the initial freeze and this audit in the fixed freeze. Test the serialization roundtrip, rerun all software checks, and require every sequential baseline to match before generating interleaved outcomes.
