# Literature options for the observed recovery failures

Primary papers checked 19 October 2026. Recommendations below are engineering
judgments for this project, not claims that these methods will solve our attacks.
No active/frozen runner was changed and no additional training was launched.

| Source | Verified mechanism | Fit and limitation in this project |
|---|---|---|
| [FLShield, IEEE S&P 2024; author paper](https://arxiv.org/pdf/2308.05832), sections 4.3, 5.2--5.3 and appendix 1.5 | Representative models, class-wise loss impact, missing-report imputation and filtering suspicious validators before model selection | Closest validation comparator and important novelty overlap. Its published n1=10 sample threshold is unsupported by our present FEMNIST private splits; see coverage audit below. |
| [FLAME, USENIX Security 2022](https://www.usenix.org/system/files/sec22-nguyen.pdf), sections 4 and 7.3 | Clustering, clipping and calibrated Gaussian noise against backdoors | Useful targeted-attack comparator beyond clean utility alarms. Requires a benign majority among participating updates; 50% corruption is outside that strict-majority assumption. Clustering/clipping alone is not a faithful FLAME implementation. |
| [FLDetector, KDD 2022](https://arxiv.org/pdf/2207.09209), sections 4.1--4.2; [official code](https://github.com/zaixizhang/FLDetector) | Predicts updates using historical information and L-BFGS; accumulates normalized prediction residuals and clusters suspicion scores | Adds evidence about individual contributors. Our 40% participation requires explicit missing-history handling and testing under writer drift. An EMA/cosine proxy must be labeled an adaptation, not FLDetector. |
| [FedRecover, IEEE S&P 2023; author paper](https://arxiv.org/abs/2210.10936) | Recovery after malicious clients are detected, using historical updates, estimated updates and exact-update corrections | Demonstrates why restoration and continued poisoned participation are separate problems. Requires suspect identification and substantially more recovery machinery than a pointer change. |
| [Crab, author paper](https://arxiv.org/pdf/2401.08216), section IV | Selective history, adaptive rollback informed by identified malicious contributions, and recovery training | Motivates a checkpoint archive rather than assuming the preceding window is sufficient. Client identification and assumptions underlying its bound remain necessary; checkpoint search alone is not Crab. |

## Data support comes before another full experiment

Read-only results are in
`artifacts/literature-fit-audit/class-validation-coverage.json`. They record SHA256
of training-label arrays and natural split records for development seeds 11/17/23,
without loading official test arrays. Coverage is measured at minimum class counts
1, 2, 5, 10 and 11. The FLShield appendix uses n1=10 and its pseudocode tests
class count >n1; both boundary conventions are audited. These counts must be
inspected before selecting that baseline or a class-wise adaptation. Lowering
the threshold changes the estimator's reliability and must be disclosed/tested;
it does not fix absent classes. Never manufacture missing validation scores or
draw validation examples from held-out tests.

## Proposed sequence, subject to diagnostic evidence

1. Finish the already frozen V5 pilot and retain all immediate/delayed negative
   results. Its aggregate paired-loss signal still does not implement FLShield.
2. Audit class coverage on the existing Fashion-MNIST and CIFAR-10 training-only
   development splits. Where supported, prioritize a faithful FLShield baseline
   and isolate any class-wise adaptation in a separate version. A simplified
   class score is not the full representative-model/report-filtering algorithm.
3. Add a faithful FLAME targeted comparator if the attack remains effective and
   baseline assumptions hold. Match training, participant and attack budgets;
   give defense noise its own recorded random stream. Measure honest exclusion.
4. Evaluate an attribution route separately from restoration. FLDetector is a
   candidate with a history/partial-participation adaptation audit. Quarantine
   decisions remain fallible and need benign-drift and adaptive-evasion controls.
5. Only with a defensible suspect signal, compare historical checkpoint
   selection plus recovery training against simple restoration and retraining.
   Count extra training/storage and distinguish detector-selected from oracle
   suspect sets. Oracle identity experiments are diagnostic upper bounds only.

The committee can attest to measured benefit and the chain can enforce authenticated
votes. Neither establishes who caused poisoning. Keep automatic poisoning-based
financial penalties disabled while attribution remains unsupported. Any future
penalties for objectively provable protocol violations would be a separately
specified mechanism and threat assumption.

These are prospective steps, not an authorization to queue every comparator or
claims of implemented defenses. The most useful next addition is determined by
coverage and the completed pilot, not by the number of methods in the bibliography.
