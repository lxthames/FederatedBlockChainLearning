# V8 outcomes and implications — 19 October 2026

The bounded comparison completed 40 of 44 new cells with zero execution failures.
Four rank/trust attack cells were held after failed FEMNIST clean gates. Runtime
was approximately 15 minutes, ending 19:20 Taipei time. Nineteen older rows were
reused; all source/data/reused-artifact bindings and participant/data comparisons
verified. Source evidence: `artifacts/comparison-v8/analysis.json`.

Three-seed Fashion-MNIST development means:

| Metric | FedAvg | Rank/trust | Median | FLTrust | Paper-based FedVal |
|---|---:|---:|---:|---:|---:|
| Clean utility | 74.6% | 74.1% | 70.5% | 20.6% | 75.5% |
| Gradual utility | 26.6% | 69.7% | 63.4% | 22.8% | 69.8% |
| Targeted utility | 64.0% | 72.5% | 66.0% | 15.7% | 74.0% |
| Mean targeted ASR | 50.8% | 14.1% | 7.5% | 37.7% | 5.5% |

FedVal is the stronger observed comparator in this pilot. Its reference access
and validation budget differ from private committees, and its audited paper-based
implementation is not established as equivalent to the released author code.
FLTrust's poor clean convergence must be reported as a limitation of this setup,
not as evidence of general inferiority or a meaningful clean-matched attack win.

On FEMNIST, FedAvg/rank clean means were 53.7%/46.8%. Rank's paired clean losses
were 12.04, 6.31 and 2.16 percentage points on seeds 11, 17 and 23. The first two
exceeded the fixed five-point bound, holding both attack conditions on those
seeds. Seed 23 alone cannot supply a three-seed attack mean or transfer claim.

These findings block large confirmation of the current candidate. Recovery
benefit remains unproven. The next bounded work isolates validation behavior;
no superiority, novelty, or broad robustness claim follows from beating FedAvg.

## Clean-metric denominator clarification

Earlier `honest_update_exclusion_rate` summaries exclude clients designated as
potential attackers, even when the run's attack is `none`. Those role-conditioned
metrics remain preserved. The V9 diagnosis adds `benign_update_exclusion_rate`,
counting every update in a clean run; it must not be silently pooled with the
older denominator. Clean rank/trust benign-update exclusions were 19.2–19.8% on
Fashion-MNIST and 22.0–23.4% on FEMNIST. This is a descriptive metric correction,
not a change to completed model training or voting.

The clean audit finds higher approval for updates with greater label-histogram
overlap between proposer training and validator validation data. It does not
establish causation or independent-vote statistical significance. See
`artifacts/bias-v9-pilot/diagnosis.json` for source hashes and per-seed values.
