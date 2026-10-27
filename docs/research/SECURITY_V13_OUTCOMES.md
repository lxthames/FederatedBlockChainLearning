# V13 security pilot outcomes

20 October 2026. All 72 cells completed with zero failures or holds. CUDA
execution on the RTX 2080 Ti ran from 05:58 to 06:46 Taipei (about 48 minutes).
All completion seals, frozen inputs, method comparisons and pre-onset condition
checks verified. All twelve per-seed attacks were effective against matched
FedAvg according to the frozen criteria. No new experiment was launched here.

Seeds 11/17/23 are development-exposed. Every mean below uses all three matched
seeds. These are descriptive pilot outcomes, not independent confirmation,
significance or evidence of novelty. Source artifacts are under
`artifacts/security-v13/` and `results/security-v13/`.

## Complete means

ASR is triggered attack success on non-target examples; lower is better. Accuracy
is on ordinary untriggered development examples. A model can retain ordinary
accuracy while remaining compromised on triggered inputs.

| Dataset | Method | Clean accuracy | Gradual-attack accuracy | Targeted-attack accuracy | Mean post-onset ASR | Endpoint ASR |
|---|---|---:|---:|---:|---:|---:|
| Fashion-MNIST | FedAvg | 75.74% | 26.10% | 66.46% | 61.55% | 98.87% |
| Fashion-MNIST | Clipping | 75.56% | 50.03% | 66.65% | 38.37% | 95.04% |
| Fashion-MNIST | FedVal adapter | 75.77% | 71.23% | 74.69% | 7.05% | 8.59% |
| Fashion-MNIST | Renormalized quorum | 73.58% | 70.81% | 66.73% | 23.69% | 70.07% |
| FEMNIST | FedAvg | 64.37% | 26.98% | 57.04% | 98.54% | 99.72% |
| FEMNIST | Clipping | 63.91% | 25.24% | 62.04% | 96.48% | 99.58% |
| FEMNIST | FedVal adapter | 59.21% | 9.62% | 59.76% | 69.02% | 84.90% |
| FEMNIST | Renormalized quorum | 61.65% | 46.60% | 62.25% | 93.21% | 99.65% |

FedVal uses central class-complete reference data and a different validation
budget. It is the audited paper-based adapter, not an independently validated
reproduction of author accuracy. Its FEMNIST clean losses are 6.11 points on
seed 11 and 7.24 points on seed 23, exceeding the frozen five-point screen.
Therefore the automated advantage screen against this FEMNIST reference is
marked uninterpretable, even where a raw numerical difference favors a method.
All raw outcomes remain reported.

## What the pilot establishes

The candidate passes all six clean cells. Mean clean accuracy deficits relative
to FedAvg are 2.16 points on Fashion-MNIST and 2.72 points on FEMNIST. Clean honest
update exclusion is 16.22% and 9.83%, respectively. This is a bounded clean screen,
not equivalence or a certificate of fairness.

Under the measured gradual reversed-update attack, candidate accuracy exceeds
clipping by 20.79 points on Fashion-MNIST and 21.36 on FEMNIST. It passes the
prespecified descriptive benefit screens against clipping and FedAvg. On
Fashion-MNIST the FedVal adapter is 0.41 points higher; there is no measured
candidate superiority over that adapter in this condition.

Under targeted model replacement, the candidate reduces Fashion-MNIST mean ASR
by 14.68 points relative to clipping but remains substantially worse than FedVal:
23.69% versus 7.05% mean ASR, and 70.07% versus 8.59% endpoint ASR. The candidate's
mean does not adequately describe the late-round compromise.

On FEMNIST, candidate mean ASR is 93.21%, only 3.27 points below clipping, failing
the frozen five-point ASR-reduction screen against that control. Endpoint ASR is
99.65%, with individual seeds 99.47%, 99.61% and 99.88%. The 62.25% ordinary
accuracy does not establish backdoor resistance. The candidate does pass the
minimal screen against weaker FedAvg, illustrating why that comparison alone
is insufficient to substantiate a broad robustness claim.

The candidate accepts 95.37% of malicious FEMNIST targeted updates after attack
onset. Its corresponding all-round honest-client exclusion is 6.67%. Fashion
targeted malicious-update acceptance is 74.44%, and all-round honest exclusion
is 10.87%. These rates describe the trajectories; they do not identify a unique
causal explanation or a safe threshold change.

## Decision

Do not promote the current candidate as a generally robust defense or launch
full confirmation on the basis of these results. It has a measured advantage
against this gradual attack, but targeted protection is inadequate, particularly
on heterogeneous FEMNIST. Keep endpoint ASR and honest exclusion visible.

A useful next diagnosis would inspect accepted targeted updates and paired-loss
votes over time, especially the private clean-data objective's inability to
detect trigger behavior. Existing trace data can establish acceptance patterns;
new counterfactual model evaluations require a separately frozen diagnostic
because full update tensors were not archived. Avoid adapting thresholds on
these repeatedly examined development outcomes. Any redesigned defense would
need fresh clean and effective-attack checks before untouched-seed confirmation.

The current evidence also supports a narrower failure-mechanism account linking
local validation, training/architecture dependence and aggregation damping.
Its novelty and publication claim still require an explicit literature audit.
