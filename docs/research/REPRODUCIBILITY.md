# Reproducibility Guide

## Environment

Create a fresh Python environment and install:

```powershell
python -m pip install -r requirements-research.txt
```

Run the test suite:

```powershell
python -m pytest -q
```

Run the synthetic two-layer smoke experiment:

```powershell
python -m scripts.research.cli run --config configs/smoke.yaml --output results/smoke
```

The output directory contains `events.jsonl` and `summary.json`. Use a new output directory for every run.

## Export frozen matrices

```powershell
python -m scripts.research.cli export-matrix --matrix core_confirmation --output artifacts/core.json
python -m scripts.research.cli export-matrix --matrix identity_committee_sweep --output artifacts/identity.json
python -m scripts.research.cli export-matrix --matrix sift_case_study --output artifacts/sift.json
```

Execute a manifest and resume safely after interruption:

```powershell
python -m scripts.research.cli run-manifest --manifest artifacts/core.json --output-root results/core --resume
```

`batch-summary.json` records completed, skipped, and failed cells. Failures are
written to `failures.json`; successful experiment directories are never rerun
when `--resume` is supplied.

Aggregate completed seed-level summaries only after all five confirmatory seeds exist:

```powershell
python -m scripts.research.cli analyze --results-root results/core --output artifacts/core-analysis.json --require-confirmatory-seeds
```

## Dataset preparation

Fashion-MNIST and CIFAR-10 are downloaded by `torchvision` into the selected data root. SIFT expects `SIFT1B_train.npy` plus `gt_<N>M_client_counts_skewed.npy` for the requested client count.

The repository currently contains targets only for 4 and 8 clients. Publishable 20-, 50-, and 80-client SIFT experiments remain blocked until shard assignments and precomputed nearest-neighbor indices are supplied. Convert those indices into targets with `routing_targets_from_neighbors`; never interpolate the existing 4/8-client labels.

## Artifact rules

- Preserve every config, raw JSONL event log, summary, software version, and seed.
- Never overwrite a completed run.
- Do not tune thresholds on confirmatory seeds.
- Generate tables and figures from raw logs, not manually copied values.
