# V20 novelty screen and claim limits

21 October 2026. Primary publication pages/author abstracts checked during this
turn. This is a focused overlap screen, not a completed systematic literature
review or implementation-fidelity audit. Published performance numbers from
other protocols are not comparable to our FEMNIST values.

| Prior work | Established overlap | What our study would need to add |
|---|---|---|
| [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) | Teacher-guided distillation | Temperature KL itself cannot be the novelty |
| [NAD, ICLR 2021](https://arxiv.org/abs/2101.05930) | Small-clean-subset teacher-guided backdoor repair, using intermediate attention alignment and a separately fine-tuned teacher | Distinguish frozen poisoned-parent output anchoring and local admission effects; compare fairly with a faithful NAD adapter |
| [Fine-Pruning](https://arxiv.org/abs/1805.12185) | Fine-tuning/pruning tradeoffs in backdoor removal | Freezing layers is not the same as pruning, but neither supports a novelty claim without a specific demonstrated effect |
| [FLIP, ICLR 2023](https://arxiv.org/abs/2210.12873) | Trigger reverse engineering and benign-client hardening in federated learning | Distinguish limited fixed-bank repair plus independent admission; no transfer of FLIP's guarantees to V20 |
| [CrowdGuard](https://arxiv.org/abs/2210.07714) | Private client feedback, hidden-neuron behavior, iterative pruning and robust feedback aggregation | Merely adding private votes or model behavior checks is not new; validate the proposed admission-specific mechanism |
| [DifFense](https://arxiv.org/abs/2202.11196) | Differential testing and MAD outlier detection against federated backdoors | Perturbed-vs-clean behavior testing is established; characterize the incremental benefit and limits of repair/admission |
| [Logit Anchoring on Clean Data, ICLR 2022](https://arxiv.org/abs/2109.01300) | Clean behavior consistency can be preserved while injecting a backdoor | Clean anchoring or good clean utility alone cannot certify absence of a backdoor; retain direct attack evaluation |

The strongest plausible contribution is a reproducible finding about **the
interaction between private validation and repair** under heterogeneous data,
with a mitigation that improves the final admitted model. It is not the first
use of distillation, frozen layers, clean anchoring or patch hardening.

V19 already supplies a narrow controlled ordering contrast: changing the order
of identical client updates can improve admission while worsening mean raw
trigger removal. It does not prove the proposed neuron, gradient dilution,
small-sample forgetting or saturation explanations. A generalizable mechanism
would need a controlled intervention on repair/validator distribution support,
sample size or decision behavior, with independent confirmation. Those
experiments are not silently added to V20.

V20 can answer whether this particular loss bundle helps at the fixed budget.
The anchored_patch versus anchored_suppress contrast isolates the incremental
penalty within the frozen-convolution setting. The baseline versus anchored
arms changes multiple factors; a positive result cannot attribute improvement
solely to KL or convolution freezing. No ablation arm is retrospectively
promoted to the predeclared primary candidate.

Novelty status: unresolved. A complete defense claim still requires online
repeated attacks, retained clean/gradual learning, fresh exposure-audited
confirmation, held-out triggers, adaptive attacks, and strong baselines.
Even a successful V20 checkpoint study would not establish acceptance readiness.
