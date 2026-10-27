# Bounded V7 development replication — frozen before execution

## Purpose and scope

Replicate the promising V6 rank/trust validation candidate on development seeds
17 and 23. Reuse the completed seed 11 controls without rewriting their artifacts.
All three seeds have been exposed to development searches; their training pools
overlap. These are development replications, not new independent confirmation.

Use the frozen V6 implementation unchanged. Fashion-MNIST, 50 clients, 40%
participation, alpha 0.5, 30 rounds, one epoch, batch 64, SGD 0.01/momentum 0.9,
30% attackers and onset 11 remain unchanged. Use prepared training-only
development holdouts. No official test arrays are read.

## Manifest: 13 new training runs

- Seeds 17 and 23: FedAvg and rank/trust under clean, gradual poisoning, and
  targeted replacement (12 runs). Clean cells precede attack cells.
- Seed 11: one clean coordinate-median control matching the V6 partial FLDetector
  adaptation's aggregation and training settings (one run).
- Reuse six seed 11 FedAvg/rank results and its clean detector result (seven rows).

The clean detector quarantined no clients, yet lost utility relative to FedAvg.
The median control tests whether aggregation explains that difference. This
single clean control cannot establish attack-detection effectiveness. Existing
held detector attack cells remain held.

## Prespecified review criteria

Preserve V6 per-seed clean requirements: FedAvg utility at least 20%, candidate
loss at most five percentage points, at most three unsupported rounds, and zero
honest monetary penalty. Hold each method/seed's attack cells if its clean cell
fails or is missing; no post-hoc gate relaxation.

Require complete three-seed matched groups, identical participants and data
fingerprints within each comparison, and effective attacks against each FedAvg
control. Gradual validity: at least one point final utility damage relative to
clean. Targeted validity: mean ASR at least ten points above clean trigger rate.
Review requires no attack utility loss greater than five points in any seed,
zero honest monetary penalties, and mean benefit at least one utility point
(gradual) or five ASR points (targeted). These are screening criteria, not
statistical significance tests or a guarantee of generalization.

Report each seed, all failed/held cells, honest exclusions, recoveries, runtime,
and mean post-onset targeted ASR. Do not mix endpoint and mean ASR. Rank/trust
has recovery disabled, so its results cannot validate accountable recovery.

Source, manifest, policy, data, and reused artifact hashes are frozen before
training. An OS lock prevents duplicate coordinators. Completed cells are reused
only after configuration validation; interrupted folders are preserved and that
cell is rerun. No automatic confirmatory expansion or threshold search.

## Launch and monitoring

Run `.venv-trust/Scripts/python.exe -m scripts.data_preparation.run_rank_replication_v7`
from the repository root. Windows background launch must use a hidden window and
redirected logs. Status: `artifacts/rank-replication-v7/status.json`; analysis:
`analysis.json` and `comparison.md` in that directory. Results:
`results/rank-replication-v7/`. Runtime is expected to be tens of minutes based
on V6; this is an estimate and GPU contention can change it.
