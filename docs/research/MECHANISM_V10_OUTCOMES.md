# V10 diagnostic outcomes - 20 October 2026

The frozen study completed all 42 clean development runs with zero failures,
finishing at 01:34 Taipei on October 20 after approximately 42 minutes. Training
used the NVIDIA GeForce RTX 2080 Ti. All 42 completion seals were verified, and
source/policy/data still match the frozen manifest. All generated pairing checks
pass. There were no unsupported rounds. No further experiment was launched.

Sources: `artifacts/mechanism-v10/{status,analysis,environment,frozen-inputs}.json`
and sealed summaries under `results/mechanism-v10/`.

## Complete three-seed descriptive means

Seeds 11, 17 and 23 are development-exposed. Fashion-MNIST uses 30 rounds;
FEMNIST uses 40 rounds and natural writer clients. Evaluation groups are fixed
across interventions. These are clean-data diagnostics, not attack experiments,
independent confirmation or statistical significance claims.

| Arm | Fashion-MNIST accuracy | FEMNIST accuracy | FEMNIST benign-update exclusion |
|---|---:|---:|---:|
| FedAvg | 74.63% | 53.68% | 0.00% |
| Clipping only | 74.51% | 53.67% | 0.00% |
| Private validation, dynamic trust | 74.14% | 46.85% | 22.63% |
| Redistributed validation, dynamic trust | 74.18% | 50.85% | 20.92% |
| Private validation, fixed trust | 73.21% | 49.28% | 20.88% |
| Redistributed validation, fixed trust | 74.96% | 50.14% | 19.96% |
| Private validation, dynamic weights, fixed eligibility | 73.64% | 47.63% | 20.75% |

Redistribution uniformly reallocates the same validation-only pool while preserving
per-validator example counts. It changes data-access assumptions and is not a
deployable private-validation solution. Dynamic diagnostic arms retain explicitly
selected historical rank-rejection penalties; they do not measure the new
runner's default no-rank-penalty policy.

## What the controlled comparisons establish in this pilot

On FEMNIST:

- Redistributing validation with dynamic trust increases mean utility by **4.00
  percentage points**, with positive effects on each seed. Validation allocation
  matters under the matched training and evaluation conditions.
- Fixing all trust channels under private validation increases mean utility by
  **2.43 points**, also positive on every seed.
- Fixing eligibility alone while retaining dynamic weights/updates increases mean
  utility by **0.78 points**. Moving from that arm to fully fixed trust adds
  **1.65 points**. The observed trust effect is not explained solely by losing
  voting eligibility. These are sequential intervention contrasts, not a unique
  additive causal decomposition across all possible channel interactions.
- Private ranking with fully fixed trust still trails clipping-only by **4.39
  points**. Removing trust dynamics does not solve the filtering problem.
- The best ranked arm, redistributed/dynamic, remains **2.83 points below FedAvg**
  on average. Its seed-11 deficit is **6.62 points**, exceeding the historical
  five-point clean-harm screen. Every ranked arm exceeds that screen on seed 11.

Lower-tail outcomes also remain concerning. Mean lower-decile client accuracy
is 33.33% for FedAvg, 30.14% for private/dynamic and 30.42% for redistributed/
dynamic. The four-point overall improvement does not establish comparable
improvement for the lowest-performing clients.

On Fashion-MNIST, dynamic trust with private validation improves accuracy by
0.94 points relative to fixed trust, the opposite direction to FEMNIST. Validation
redistribution with dynamic trust changes the mean by only +0.04 points. The
redistributed/fixed arm reaches 74.96%, just +0.33 points over FedAvg; this is not
evidence of a significant advantage. The effect is dataset/configuration dependent.

Private/dynamic FEMNIST ends with 6, 8 and 5 clients below the validator trust
threshold on seeds 11, 17 and 23. Nevertheless all runs maintain functioning
committees: no committee-unavailable events or insufficient-response decisions
were observed in these honest-validator clean runs. Do not describe this pilot
as demonstrating a liveness collapse. The earlier adversarial-withholding
reproduction is separate evidence.

## Decision

Do not promote any ranked arm to a large confirmatory study yet. This pilot
supports a narrower mechanism claim: validation allocation and trust feedback
both affect clean utility under natural client heterogeneity, and a substantial
ranking-related loss remains even with fixed trust. It does not show that trust
is universally harmful or that any mitigation is robust against poisoning.

The next bounded design should address relative rank rejection itself and its
coupling to trust, retaining clipping and audited FedVal controls. Fix the rule
and its evaluation criteria before execution; require clean preservation before
testing effective gradual and targeted attacks. Clipping's clean result alone
cannot establish security, as the earlier V9 targeted pilot showed. Preserve
these negative findings in the paper's evidence trail.
