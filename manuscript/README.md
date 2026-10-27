# Manuscript and reproducibility package

The paper reports the frozen 400-cell `lifecycle-v25` main study and 200-cell
`lifecycle-v26-coordinate` companion study separately. Repeated controls are not
additional independent seeds. The contribution is an empirical lifecycle
comparison of repair, admission, and subsequent training.

## Build the paper from saved results

Run these commands from the repository root in PowerShell. Use the project
interpreter: the global Python environment is missing training dependencies.

```powershell
.venv-trust/Scripts/python.exe manuscript/audit_evidence.py
.venv-trust/Scripts/python.exe manuscript/make_diagnostics.py
.venv-trust/Scripts/python.exe manuscript/make_figures.py
Push-Location manuscript
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
Pop-Location
.venv-trust/Scripts/python.exe manuscript/verify_manuscript.py
```

`audit_evidence.py` recomputes sample counts and compares shared controls. It
writes `data/evidence_audit.json` and `sections/sample_counts.tex`, checks both
frozen source snapshots against the live source, and validates per-run endpoint
and horizon summaries. These commands check saved outputs; training commands
are provided separately below.
The complementary full companion-study audit is:

```powershell
.venv-trust/Scripts/python.exe scripts/audits/audit_lifecycle_v26_coordinate.py
.venv-trust/Scripts/python.exe -m pytest -q tests/test_lifecycle_v25.py
```

The diagnostics generator creates the policy, phase, candidate, and gate tables from sealed saved outputs. The
figure generator uses the manifest-selected completed histories. The companion comparison and paired-contrast tables are checked
against the frozen companion analysis by `verify_manuscript.py` (see below).

## Final experimental artifacts

- Main manifest and analysis: `artifacts/lifecycle-v25/{manifest,analysis}.json`.
- Companion manifest and analysis:
  `artifacts/lifecycle-v26-coordinate/{manifest,analysis}.json`.
- Each completed output under the corresponding `results/` directory has a
  configuration, split manifest, event stream, summaries, final model, and
  `completion.json` file-hash seal. Repair trajectories also retain candidate,
  source, committed-model, decision, and offline-evaluation artifacts.
- `frozen-inputs.json` binds source, data, protocol, launcher, and test hashes;
  each job's `source-snapshot/` preserves its frozen implementations.
- The final analysis uses 400 completed main-study trajectories and 200 completed
  companion trajectories. Results and paired contrasts are computed within each
  study; repeated controls retain five-seed sample sizes.
- Repeated Fashion-MNIST controls differed across executions, limiting exact
  repeatability.

## Dataset identity and counts

The 62-class FEMNIST data come from the historical TFF HDF5 archive:
`https://storage.googleapis.com/tff-datasets-public/fed_emnist.tar.bz2`.
Archive SHA-256:
`fe1ed5a502cea3a952eb105920bff8cffb32836b5173cb18a57a32c3606f3ea0`.
`data/femnist-tff/provenance.json` records the raw HDF5 hashes and the upstream
loader revision `a484b84e053dca1a1d6afa29b86a3cd5ece9dd9d`.

`data/lifecycle-v25/provenance.json` records the historical exposure audit and
five new writer/reference cohorts. Within a seed, train/test examples come from
the same enrolled writers. Cohorts are disjoint from historical experiments and
across seeds. Exact pixel matches to the evaluation images are removed from
training and reference pools. Original float32 [0,1] inverse-background pixels
are retained and a channel dimension is added; there is no learned normalization.
`sections/sample_counts.tex` lists training, private-validation, test, and ASR
sample counts. The 620 reserved reference examples are not the private repair
allocation: repair and admission use nonparticipating clients' validation data.

Fashion-MNIST uses its official 60,000/10,000 train/test split, float32 pixels
scaled by 255, 50 Dirichlet-alpha-0.5 clients and a 20% private-validation split.
Its non-target test denominator is 9,000. Official images had prior development
exposure. Exact prepared arrays are bound by `frozen-inputs.json` data hashes.

## Environment

All 600 saved environment events record NVIDIA GeForce RTX 2080 Ti, Python
3.13.9, PyTorch 2.11.0+cu126, NumPy 2.4.2, and CUDA execution. The runner enables
PyTorch deterministic algorithms, disables cuDNN benchmarking, sets
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, and uses two CPU threads.

Inspection of the project environment on October 25, 2026 additionally found
Windows 11 build 26200, torchvision 0.26.0, h5py 3.15.1, hdbscan 0.8.44,
SciPy 1.18.1, scikit-learn 1.9.0, pytest 9.0.3, matplotlib 3.11.2, CUDA runtime
12.6, and cuDNN 91002. These additional versions describe the inspected environment;
individual historical runs recorded only the versions listed above. Fresh setup requires a matching
CUDA-enabled torch/torchvision pair plus the listed dependencies; the older
requirements files alone omit hdbscan and other transitive dependencies.

## Independently rerun one training cell

Do not use the historical launcher in place: it can coordinate other jobs and
reuse completed outputs. The wrapper below redirects only the output directory,
checks frozen source/data, and refuses existing destinations. It does not alter
training code, seeds, or frozen results. A real run requires a CUDA device and
can take substantial time; it is distinct from rebuilding the manuscript.

```powershell
# Inspect the exact frozen cell without writing files or running training:
.venv-trust/Scripts/python.exe manuscript/reproduce_cell.py --study main --index 0 --output-root C:/Temp/lifecycle-replication-main-0 --describe
# Execute that cell into a new directory:
.venv-trust/Scripts/python.exe manuscript/reproduce_cell.py --study main --index 0 --output-root C:/Temp/lifecycle-replication-main-0
# Companion cell indices are defined in its separate manifest (0..199):
.venv-trust/Scripts/python.exe manuscript/reproduce_cell.py --study companion --index 0 --output-root C:/Temp/lifecycle-replication-companion-0
```

For the complete matrices, enumerate main indices 0..399 and companion indices
0..199 with a distinct new output root for each invocation. Record the
environment for each run and compare new `summary.json`, `lifecycle-summary.json`, and model hashes with the matching
frozen cell. CPU checks of saved checkpoints validate inference; the commands above
execute training.

## Presentation and verification

Plots are generated at their intended printed width, with distinct markers and
line styles. Utility and admission plots remain auxiliary artifacts; their
numbers are already included in the policy table. All substantive results remain
in the paper. `verify_manuscript.py` checks table values, intervals, references,
and PDF pagination/font embedding after compilation. Review rendered pages as
well as these automated checks. The GenAI Usage Disclosure describes assistance with manuscript editing,
figure and table scripts, and checks of saved experimental artifacts.
