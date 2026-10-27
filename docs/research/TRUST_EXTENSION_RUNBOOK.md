# Versioned TRUST extension

## Current execution: GPU

The author authorized transfer to the RTX 2080 Ti on 19 October 2026. The CPU
coordinator is stopped and its completed/partial artifacts are preserved.
The authoritative study is now `artifacts/trust-extension-v1-gpu/status.json`,
with results under `results/trust-extension-v1-gpu/`. The launcher is
`scripts/data_preparation/run_extension_gpu.py`. It reuses the exact frozen
protocol modules and checks their hashes together with the launcher itself.
All 308 configuration comparisons were checked: only device and experiment ID
change. GPU development gates and evaluation seeds run afresh; CPU seed results
are not pooled into GPU groups. The manuscript artifact generator selects the
GPU study once its inputs are frozen.

Monitor with:

```powershell
Get-Content artifacts/trust-extension-v1-gpu/status.json
Get-Content artifacts/trust-extension-v1-gpu/worker.stderr.log -Tail 20
```

If interrupted, with no live GPU coordinator, restart using:

```powershell
.\.venv-trust\Scripts\python.exe -m scripts.data_preparation.run_extension_gpu
```

The launcher requires CUDA and refuses CPU fallback. Its runtime forecast is in
the new job directory and supersedes the historical CPU estimate below. Both
classification jobs share the GPU, so actual completion time depends on their
concurrent workloads. SIFT remains on CPU. The original CPU execution notes
below document the preserved historical run and are not restart instructions
for the active GPU study.

Implemented 19 October 2026. The implementation is under
`scripts/trust_extension/`; it uses an isolated copy of the original research
modules so active v2 and SIFT jobs retain their fingerprints. Run this study
through `scripts.trust_extension.study`, not the legacy command-line entry
points retained in its source snapshot.

## Prepared data

`data/femnist-tff/provenance.json` records the official 62-class archive hash,
raw HDF5 hashes, eight seed-specific writer partitions and generated-file hashes.
Each seed selects 50 enrolled writers and distinct reference writers. Official
tests are preserved. Training/reference pixels are globally deduplicated,
including matches to tests. No test labels enter that preprocessing decision.
The pipeline uses HDF5 directly; TensorFlow installation is unnecessary.
Install additional dependencies with:

```powershell
.\.venv-trust\Scripts\python.exe -m pip install -r requirements-trust-extension.txt
```

The official archive checksum is hard-coded from the historical TFF loader and
must pass before extraction. Pixel background is white (one); the backdoor trigger
uses black (zero). Preparation fails on insufficient surviving data or reference
class coverage. Per-writer held-out utility is reported with test counts.

## Study design

| Stage | Development/evaluation seed runs |
|---|---:|
| Development: FedAvg attack controls, FedVal/full clean checks, adaptive controls | 33 |
| Matched five-method comparison, Fashion-MNIST and FEMNIST, three conditions | 150 |
| Adaptive proxy-based model replacement, five methods | 25 |
| False-recovery reports and vote withholding | 40 |
| Continue vs scheduled restoration vs scheduled restoration with sanctions | 30 |
| Uniform participation, full extension and Multi-Krum | 30 |
| Total evaluation cells before gates | 275 |

Evaluation seeds: 13,29,47,71,101. Development: 3,5,7. Defaults: 50 clients,
40% participation, 100 rounds, one epoch, learning rate 0.01. Development uses
20 rounds. FEMNIST has natural writer distributions; Fashion-MNIST uses
Dirichlet alpha=0.5. CPU two-thread execution avoids additional GPU contention.
Attack gates require three completed clean controls and matched attack controls.
The fixed clean-utility feasibility floor is 0.10 for Fashion-MNIST and twice
uniform chance (2/62) for FEMNIST. Untargeted attacks need mean utility loss
above .005; targeted attacks need mean ASR lift over clean-trigger controls
above .05. Ineffective/incomplete development conditions are held, not relabeled
as defense success. These gates are development diagnostics, not confirmatory
claims. Failed cells are retained.

The new recovery quorum guard is a protocol change. Do not merge extension
results with v2 outcomes. The original core cannot validate the guard. Before
submission, choose the final protocol explicitly and report dataset/baseline
coverage for that version; if a claim needs final-version CIFAR-10 or BOBA
evidence, those runs are still required. This limitation cannot be closed by
pooling results from different versions.

## Background execution and monitoring

The study was launched hidden in the background. Its authoritative paths are:

- `artifacts/trust-extension-v1/status.json`: actual worker PID, stage, counts.
- `worker.stdout.log`, `worker.stderr.log`: execution logs.
- `manifest.json`, `frozen-inputs.json`: immutable configuration/code/data inputs.
- `attack-validation.json`, `held-experiments.json`, `failures.jsonl`: explicit gaps.
- `systems-failures.jsonl`: real-EVM replay failures, separately counted.
- `five-seed-analysis.json`, `completeness.json`: complete groups and paired results.
- `runtime-forecast.json`: preliminary scheduling estimate and assumptions.
- `results/trust-extension-v1/`: config, splits, events, summaries and traces.

The initial forecast is about 83--167 hours based on six concurrent one-round
preflight timings, before accounting for held runs. It is a rough planning range,
not a promised completion time. Those preflight files predate pixel deduplication
and remain excluded from scientific tables. Runtime and communication conclusions
need their stated limits: concurrent wall time and estimated payload bytes.

Read progress with:

```powershell
Get-Content artifacts/trust-extension-v1/status.json
Get-Content artifacts/trust-extension-v1/worker.stderr.log -Tail 20
```

If interrupted, rerun the exact frozen code using:

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:OPENBLAS_NUM_THREADS='2'
.\.venv-trust\Scripts\python.exe -m scripts.trust_extension.study run
```

Completed cells require matching config and frozen inputs. Partial directories
are preserved with an interrupted suffix, then the whole cell restarts. Do not
launch a second coordinator against the same output root while the first lives.
Do not edit `scripts/trust_extension/` during execution; any needed change requires
a fresh version and output root. The worker validates source hashes between runs.

The first full-framework development trace is replayed on the persistent local
Hardhat node. The adapter checks candidate decisions and active pointer per round,
then final stake agreement, retaining actual receipts. This demonstrates equivalent
execution on identical decisions; it does not establish that EVM execution improves
statistical robustness or outperforms a signed centralized log.

## Analysis and manuscript

Only complete five-seed groups enter means and seed-bootstrap intervals. Paired
utility comparisons and recovery-only controls retain seed correspondence.
Recovery-only pre-intervention histories are asserted identical. Five seeds give
limited precision; intervals alone are not significance claims.

The paper build now invokes `scripts.data_preparation.extension_artifacts` to
generate CSV and TeX from complete groups. An entire matched dataset table needs
all 15 method/condition groups; partial comparisons remain visibly pending.
Rebuild and render with the normal manuscript build, then inspect all pages and
the nine-page budget. Publication readiness remains false pending complete
evidence, source/claim review, protocol reconciliation and author review.

See [baseline/protocol audit](TRUST_EXTENSION_BASELINE_AUDIT.md) for FedVal
differences, the strengthened recovery guard and the precise attacker knowledge.
