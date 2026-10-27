# Full TRUST matrix restart, 22 October 2026

Run: `scripts/run_trust_full.ps1`; status: `artifacts/trust-full-20261022/status.json`.
Results: `results/trust-full-20261022`. This run recomputes all classification
cells with a single code version. Earlier results remain in their original
directories. Exact source hashes are required for resume and paired analysis.
The source snapshot, source hashes and manifests are recorded before training.

| Stage | Trajectories |
|---|---:|
| Development | 60 |
| Calibration | 27 |
| Core comparison | 300 |
| Client population | 60 |
| Data heterogeneity | 80 |
| Attacker fraction | 60 |
| Identity and committee | 80 |
| Ablations | 60 |
| Benign drift | 45 |
| Classification total | 772 |
| SIFT supporting case study | 180 |

Fashion-MNIST and CIFAR-10 are used in the core study. Follow-up sweeps use
Fashion-MNIST. Existing development seeds are 3/5/7; main seeds are
13/29/47/71/101. These seeds and datasets were previously exposed, so this is
a consistent rerun of the existing protocol, not fresh independent confirmation.
Core and sweep runs use 100 rounds; development and calibration use 20 rounds.
Attack validation on development data can hold downstream cells; such exclusions
are recorded in held-experiments.json, never counted as successful runs.

ALIE uses the mean plus one population standard deviation of available honest
reference updates. When trust filtering leaves fewer than two such references,
the previous implementation crashed. The current operational policy submits the
unmodified local update and records attack_applied=false and the reference count
in each round, with cumulative alie_reference_shortfalls in the summary. This
changes the attack protocol; old and new results must not be combined. Attack
presence metrics describe participating malicious identities and should not be
interpreted as proof that every submission was perturbed. Report shortfall counts
alongside ALIE outcomes.

The SIFT stage requires verified query/target correspondence and normalization
in data/sift-provenance.json. It is explicitly held when this is absent; the
available legacy arrays alone do not establish correspondence.

Analysis now groups by both configuration and source. Incomplete source cohorts
are reported and excluded from five-seed estimates. Reporting errors are recorded
without preventing later training stages from executing. Cell failures are
recorded and execution proceeds to remaining cells. Final status is
completed_with_gaps if any cell failed, was held, or analysis failed.

This matrix covers the existing TRUST robustness protocol. The proposed lifecycle
article still needs independent repair baselines, held-out triggers and evaluation
splits described in TRUST_2027_FULL_ARTICLE_ASSESSMENT_2026_10_22.md. The 27-cell
V24 mechanism study is complete and remains development evidence. Completing this
matrix alone does not establish publication readiness for that proposed article.
