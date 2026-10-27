# V11 architecture and local training audit

20 October 2026. The audit inspects the frozen V11 source and uses a separate
CPU diagnostic on enrolled seed-11 training examples. No V11 source, policy,
data, or completed result was edited. The existing GPU worker continues.

The current evidence does not identify a broken CNN or establish architecture
as the main cause of the defense gap. It identifies unequal local optimization
as a substantial contributor to the client-size/update-magnitude association.
Architecture may interact with that effect; an architecture intervention has
not yet been run.

## Which model actually runs

`scripts/trust_screen_v11/study.py` passes `model='default'` to the factory in
`scripts/trust_screen_v11/research/models.py`. Both classification tasks use:

```text
Input: 1 x 28 x 28
  Conv 3x3, padding 1, 1 -> 16 channels -> ReLU -> MaxPool 2x2
  Conv 3x3, padding 1, 16 -> 32 channels -> ReLU -> MaxPool 2x2
  Flatten: 32 x 7 x 7 = 1,568
  Linear: 1,568 -> 128 -> ReLU
  Linear: 128 -> 62 FEMNIST logits / 10 Fashion-MNIST logits
```

This is a shared classifier with class outputs. It does not have per-client
prediction heads. `scripts/core/models.py::EnhancedMultiHeadDNN` belongs to
the separate `emh_dnn` routing configuration and is not used in these runs.
Any manuscript description of the experiments must make that distinction.

| Layer group | FEMNIST parameters |
|---|---:|
| First convolution | 160 |
| Second convolution | 4,640 |
| Hidden linear layer | 200,832 |
| Output classifier | 7,998 |
| Total | 213,630 |

Fashion-MNIST has 206,922 parameters. The hidden linear layer alone contains
94.0% of the FEMNIST parameters. This concentration is an architectural property,
not proof that it causes validation failures. The model has no BatchNorm,
dropout, running-statistic buffers, or final softmax.

## Compatibility and numerical checks

- Both prepared inputs are float32, NCHW, 28x28, with values in [0,1]. FEMNIST
  labels span 0 through 61, and Fashion-MNIST labels 0 through 9. The output
  dimensions match. FEMNIST's inverse background is retained consistently.
- Training uses cross-entropy on raw logits. There is no double softmax in this
  classification path. Train-mode and eval-mode logits are identical for the
  inspected batches, as expected with these layers.
- The delta helper subtracts global parameters from locally trained parameters;
  the aggregation helper adds the resulting delta. This CNN has no integer or
  floating running buffers that could confuse those operations.
- Within each development seed, all six clean V11 methods have the same initial
  model hash. The completed source/data checks match the frozen manifest.
- The independent CPU numerical check selected one enrolled training example per
  class, 62 examples total, by a fixed first-index rule. With full-batch SGD,
  learning rate .05 and momentum .9 for exactly 200 updates, training loss fell
  from 4.13053 to 0.000374 and training accuracy rose from 1.61% to 100%.
  Every inspected parameter gradient was present and finite. At 100 updates,
  accuracy was 79.03% and loss 1.28024.

The memorization result rules out gross non-learning on this small input set.
It does not establish adequate capacity, generalization, or convergence under
federated averaging, and is not an accuracy result for the paper's benchmark.

## The consequential interaction is local optimization length

For FEMNIST, every round starts each local model from the global model and creates
a new SGD optimizer with learning rate .05 and momentum .9. Momentum is reset
between client-round training calls. There is one local epoch, batch size 16,
no learning-rate schedule, no weight decay, and no local proximal penalty.

Seed-11 enrolled clients have 61 to 240 training examples, so an epoch gives
them **4 to 15 optimizer steps**. The same learning rate and epoch count do not
mean the same optimization budget. Momentum also accumulates for more steps in
the longer local runs. The subsequent aggregation weights additionally depend
on sample count, and validation/clipping acts on those local updates.

The bounded diagnostic starts all 50 clients from the same initial model and
compares the existing one-epoch training function with its first four steps.
Both use the same per-client shuffle and optimizer. For clients whose complete
epoch is four steps, the two resulting state dictionaries match exactly.

| Initial-model CPU measurement | One local epoch | Four local steps |
|---|---:|---:|
| Correlation: client sample count vs update L2 norm | +0.9551 | -0.4328 |
| Mean update norm, smallest ten clients | 0.5182 | 0.3084 |
| Mean update norm, largest ten clients | 1.4692 | 0.2701 |
| Mean update norm, all clients | 0.8913 | 0.2923 |

Thus the strong positive size association is sensitive to training length;
larger clients do not intrinsically have to produce larger updates. The negative
association under four steps is descriptive and is not an independent finding
of a beneficial size effect. This intervention also changes how many local
examples are processed. It does not isolate momentum from step count or resolve
label-composition differences.

Under the original epoch rule, the mean share of squared update norm is 9.15%
in the convolutional layers, 55.10% in the hidden linear layer, and 35.75% in
the output classifier. These are means of per-client energy proportions at
initialization, before clipping. Much of the early movement occurs after the
convolutions. This motivates inspecting classifier drift, but neither parameter
count nor update energy establishes which layer harms global learning.

The earlier V11 trace diagnosis found that scaled quorum disproportionately
rejects larger-data clients. The new step-budget probe supplies a concrete
upstream explanation worth testing: longer local optimization changes the
updates that private validators assess. It does not prove that update magnitude
alone causes rejection. In particular, a smaller loss probe can scale both its
mean change and standard error together, leaving the approval inequality
approximately unchanged in a linear regime. Direction and label support matter.

## Same architecture, substantially different clean outcomes

All three development seeds (11, 17, 23) have now completed every clean arm.
FEMNIST mean accuracy at the fixed 40-round budget is:

| Method | Mean development accuracy |
|---|---:|
| FedAvg | 53.68% |
| Clipping only | 53.67% |
| Paper-based FedVal adapter | 45.95% |
| Original rank/trust | 46.85% |
| Rank without rejection penalties | 48.99% |
| Scaled quorum | 24.65% |

The CNN can learn substantially more under the controls with the same data,
initialization and training settings. Architecture alone cannot account for the
method contrast; an interaction with validation, weighting, clipping or learning
duration can. All curves use a short development budget, so these measurements
do not establish asymptotic convergence or the best achievable accuracy.

## Recommended discriminating experiment

The next bounded clean diagnostic should hold the CNN fixed and compare the
existing one-epoch rule with a prespecified fixed-step rule across matched
FedAvg, clipping-only and scaled-quorum runs. Report processed examples and
computation explicitly: four steps changes the budget and is not automatically
a fair efficiency comparison or the preferred final setting. Retain the separate
accepted-mass versus renormalization controls proposed in the trace diagnosis;
do not change both mechanisms in one unidentifiable fix.

Then repeat a compact, fixed comparison with a second prespecified backbone to
measure architecture-by-defense interaction. A larger network alone is not yet
an evidence-based remedy. Improving the architecture could raise every method's
accuracy while leaving the clean defense deficit intact. Clean improvements
still require fresh matched attack evaluation.

No alternative-backbone or full federated training experiment was launched by
this audit. Its only new training was the 100 local CPU probes and the 200-step
small-sample numerical check described above.

## Reproduction

```powershell
.venv-trust/Scripts/python.exe -W error::RuntimeWarning -m scripts.audits.audit_screen_v11_architecture
```

Machine-readable output: `artifacts/screen-v11-architecture-audit/audit.json`.
It includes architecture counts, all 50 paired local probes, memorization
checkpoints, matched clean summaries, and the audit script's source hash.
