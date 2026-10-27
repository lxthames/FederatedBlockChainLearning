# Readiness validation — 19 October 2026

The 275-cell GPU follow-up remains stopped. Its 33 development results are
preserved. Original classification and SIFT workers continue independently.
Do not resume the follow-up solely because implementation tests pass.

## Checks completed

`TRUST_EVM_TESTS=1 python -m pytest tests -q -p no:cacheprovider`:
86 passed in 16.83 seconds, including local EVM integration checks.

Recomputed original follow-up development gates from all three seeds:
Fashion-MNIST targeted replacement and gradual poisoning pass; FEMNIST targeted
replacement passes the numerical ASR gate, but clean learning remains weak.
FEMNIST gradual poisoning fails with paired final-utility losses [0, 0, 0].
The existing clean feasibility floor does not establish useful convergence.
These gates are diagnostics, not evidence that all attacks or defenses are valid.

## Bounded background diagnostics

Launcher: `scripts/data_preparation/validate_readiness.py`.
Status/results: `artifacts/readiness-validation-v1/` and
`results/readiness-validation-v1/`.

All 18 cells use development seeds 3, 5, 7 on CUDA:

- Nine clean FEMNIST FedAvg runs: 100 rounds, learning rates .01/.05/.1,
  one local epoch. Record learning curves, majority-class test accuracy, and
  per-client training sample counts. This separates short warm-up from persistent
  non-learning; it does not select a final setting automatically.
- Nine clean Fashion-MNIST runs: 20 rounds, validation-only, no trust-based
  exclusion, and permissive-vote/no-trust-exclusion controls. Compare against
  the preserved clean FedAvg/full development results. These controls isolate
  mechanisms; permissive voting is not a proposed deployable defense.

Freeze source hashes and configurations. Preserve failures. Never read
confirmatory outcomes for tuning. No automatic confirmation launch occurs.
Changed configurations will require fresh matched attack-development checks;
successful clean runs alone cannot authorize the full matrix.

## Remaining submission blockers

1. Select one final protocol. The extension's local recovery-attestation quorum
   differs from original v2; original v2 results cannot validate that guard.
2. Cover the selected final protocol across the planned core datasets, attacks,
   baselines, sensitivity and ablations. The current extension matrix does not
   cover CIFAR-10, coordinate median, BOBA, sign flip, or ALIE.
3. Resolve or explicitly delimit ALIE applicability and failed SIFT FLTrust
   groups. Do not silently substitute implementations or drop failed seeds.
4. Establish useful FEMNIST clean learning before interpreting targeted ASR,
   then revalidate attacks under any changed configuration.
5. Complete real-data trace replay costs at 20/50/80 clients for the chosen
   protocol. Existing synthetic development profiles do not close this gate.
6. Complete five-seed analyses, claims/source audit and final rendered anonymous
   manuscript checks. The paper remains an unfinished draft.

Use the diagnostic results to decide a minimal corrected matrix. Preserve
compatible completed baseline evidence only after checking configuration,
source, data and randomness; never pool different protocol versions.
