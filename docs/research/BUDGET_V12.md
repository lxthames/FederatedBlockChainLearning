# V12: local training budget, architecture width, and aggregation mass

Prepared 20 October 2026 before execution. This is a bounded clean development
diagnostic following V11. Sources: `scripts/trust_budget_v12/`; results:
`results/budget-v12/`; coordinator: `artifacts/budget-v12/`.

## Questions and finite matrix

The initial-model CPU probe found a size/update-norm correlation of .955 with
one local epoch and -.433 with four local steps. V11 clean scaled voting rejects
larger-data clients disproportionately and leaves rejected aggregation mass
unused. The CNN learns more effectively with FedAvg/clipping than scaled quorum.
These observations motivate separate interventions; they do not establish that
changing the model or step budget will resolve validation bias.

Exactly **60 clean FEMNIST cells**: 3 exposed seeds (11/17/23) x 2 architectures
x 2 local budgets x 5 aggregation arms. The entire default-CNN matrix runs first,
followed by the width intervention. No threshold search, attack matrix, or
automatic confirmatory experiment is part of this study. All cells execute even
if a clean method performs poorly; a missing/invalid replay source holds only its
dependent damping cell. Preserve failures and report all seeds.

## Training and model interventions

Use the existing prepared training-only development data and fixed natural writer
partitions/evaluation groups. Fifty clients, 20 participants per round, original
bounded role schedule, 40 rounds, batch size 16, SGD learning rate .05, momentum
.9, local optimizer reset each client-round. All data are clean: potential attacker
identities affect the frozen participant schedule but never poison an update.

- `epoch`: existing one-local-epoch behavior, including partial final minibatches.
- `fixed_steps`: exactly four optimizer steps per client-round, using the same
  deterministic shuffle. If a client has fewer than four minibatches, repeat the
  deterministic loader with its generator continuing. Count repeated examples.
  Local epochs must be configured as one in this mode. Partial minibatches remain
  partial, so this equalizes steps rather than exact example counts.

Preserve full local training-set sizes as aggregation sample weights in both
budgets. Changing sample weights simultaneously would confound the intervention.
Four steps processes fewer examples for most writers; report actual processed
examples, optimizer steps, validation work and runtime rather than asserting an
equal-compute or equal-exposure comparison. Longer training and a momentum-only
intervention are not identified by this matrix.

- `default`: unchanged CNN, channels 16/32, 3x3 padded convolutions with ReLU
  and max pooling, flatten, 128-unit hidden linear layer, 62-class logits.
- `wide_cnn`: double convolution channels to 32/64, retain the same kernel sizes,
  pooling, 128-unit hidden layer and 62 outputs. No normalization/dropout or extra
  depth. This is one prespecified width intervention, not an author-model
  reproduction. Each architecture uses the same seed for initialization; different
  shapes necessarily have different initial weights. Every method within a
  seed/architecture starts from identical weights.

The default model has 213,630 parameters and the wider model has 428,350.

No data, optimizer, tolerance or training-round adjustments based on observed
outcomes. Architecture effects include the consequent parameter/computation
change; this does not isolate individual layer capacity contributions.

## Aggregation arms

All validation uses the same fixed private data and honest proposer-excluding
committees. Trust updates, trust weighting, trust eligibility and monetary
penalties are fixed/disabled. No recovery, EVM, or sanctions are measured.

1. `fedavg`: sample-weighted average, no clipping or filtering.
2. `clip_only`: clip at twice the round median update norm; accept all finite
   proposals and sample-weight their average. The inherited keep-all ranking
   path still computes validation, which must be counted as overhead.
3. `scaled_nominal`: V11 scaled private voting, clipping 2, paired loss change
   mean <= 2 standard errors, zero additive tolerance, at least two examples,
   and ceil(2C/3) affirmative votes. Probe each candidate at its offered nominal
   sample share. Apply accepted nominal contributions without renormalization.
4. `scaled_renormalized`: identical proposal probes, rule and committees; normalize
   accepted sample weights to sum to one. If none pass, apply a zero update.
   Its model/votes evolve along its own trajectory, so this is an online policy
   intervention rather than a replay of identical updates.
5. `clip_damping_replay`: accept all clipped proposals and multiply their ordinary
   sample-weighted average by the per-round accepted nominal mass recorded by the
   matching completed `scaled_nominal` cell (same seed, budget and model). This
   is a diagnostic exogenous schedule, not a deployable defense. Its full schedule
   and source completion-seal hash are stored in `diagnostic-inputs.json` and the
   summary. Missing or mutated source evidence fails closed. Zero mass means zero
   contribution and is not counted as a consensus failure in this control.

The 2-SE rule remains a heuristic under reused correlated private examples;
there is no calibrated global-quality, confidence, or security guarantee.

## Evidence and comparisons

Freeze package sources, this protocol, prepared data/provenance and parent V11
freeze hash. Record sealed configuration, summary, splits, events, chain trace
and diagnostic inputs. Verify dependent source seals before use, on resume and
in binding checks. Do not rewrite completed summaries. Preserve interrupted
folders under the intended result root.

Strict method comparisons require identical data, initialization, train/validation
partitions, evaluation groups, sample schedule, training settings and source.
Cross-budget comparisons permit only `local_training_budget` to change.
Cross-architecture comparisons permit only `model` and its consequent initial
model hash to change. Missing or altered evidence invalidates the comparison.

Report seed-level final/round accuracy, lower-tail writer utility, clean exclusion,
submitted-client sample count vs rejection/update magnitude, accepted nominal mass,
applied coefficient sums, optimizer steps, processed examples, validation work and
runtime. Group means require all three matched seed runs; no averaging partial
groups. Comparisons and width-by-defense/budget-by-defense differences are
descriptive within the exposed development seeds, not significance claims.

The existing per-seed screen requires FedAvg accuracy >=20%, no more than five
points of candidate accuracy loss, <=3 unsupported rounds and no honest monetary
penalty. It is displayed only as a development indicator and does not
automatically promote a method. Any claimed
clean-preserving defense must subsequently show effective attack performance.
No results from different protocol versions are pooled.

## Execution

Run tests before freezing. Background execution uses CUDA, a hidden window and
redirected logs. The coordinator waits for V11 to end successfully, checks its
lock is released, and then starts. The queue exits on an unsuccessful parent;
it does not overlap the existing V11 GPU worker.

```powershell
.venv-trust/Scripts/python.exe -m pytest tests/test_budget_v12.py tests/test_screen_v11.py -q
.venv-trust/Scripts/python.exe -m scripts.trust_budget_v12.study --freeze
.venv-trust/Scripts/python.exe -u -m scripts.trust_budget_v12.study --run --wait-for-v11
```
