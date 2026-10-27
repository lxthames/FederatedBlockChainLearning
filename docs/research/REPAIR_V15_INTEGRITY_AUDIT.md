# V15 integrity audit, October 20, 2026 (Taipei)

The V15 background coordinator was stopped after 20 completed cells (six
calibration, fourteen evaluation) when an unexplained model divergence was
confirmed. Its original status snapshot, intervention record, completed artifacts
and interrupted cell are preserved under artifacts/repair-v15 and results/repair-v15.
No V15 source, completed result, completion seal or frozen policy was rewritten.

## Decisive evidence

Fashion-MNIST seed 11, aggregate-only versus repair: parent model hashes match
through round 8. At round 9 all twenty clipped candidate hashes, their acceptance,
the screening/verification panels and behavioral vote measurements match. Both
arms accept the full aggregate; no repair has taken place. Nevertheless the
committed model hashes differ and ordinary accuracy becomes 40.70% versus 62.73%.

Instrumented and ordinary isolated replays produce the repair trajectory exactly
through round 11 (62.73% at round 9, 72.79% at round 10, 71.10% at round 11).
An additional replay archives the round-9 parent, all clipped updates, proposed
delta and expected model. Comparing it with the original aggregate-only checkpoint
at round 11 isolates exactly three mismatching parameters in 7.weight:

| Flattened index | Original committed value | Replayed expected value |
|---:|---:|---:|
| 246823 | 143.80557251 | -0.00857297145 |
| 246960 | 1457.92834473 | -0.00567753147 |
| 246961 | -128.21627808 | 0.00892879441 |

Every other model entry matches. The archived round-11 parent hashes to the
recorded original round-9 commitment, so the anomalous weights agree with the
live execution trace rather than merely a mismatched filename. The original
aggregate arm then holds all remaining 91 rounds on that corrupted model.

Observed corrupted hash:
`1893997bb32a1d31c43f1ff585907b8e988658dcd6f1e0ee496dc1523d4dc530`

Replayed round-9 hash:
`7fb64c24d3e347fc5978793e087ffc29d2633834a8509b41562cecf81f1da0f2`

## Interpretation and scope

Withdraw the interpretation of this 40.70% endpoint and 91 holds as evidence
that the aggregate-validation algorithm stalls honest learning. The discrepancy
is an execution-integrity failure. The hardware/runtime cause is unresolved;
neither a specific GPU defect nor an application source-code defect is established.
No poisoning was active in these runs. Do not infer that the algorithm is sound,
that repair works, or that other historical runs are necessarily corrupted.
The V15 repair run itself had zero successful repairs, so its clean utility still
does not demonstrate repair effectiveness. Hold V15 scientific comparisons pending
guarded replication; its old generated comparison.md is historical unqualified output.

## Evidence paths

- Instrumented 11-round replay: artifacts/repair-v15-commit-audit-aggregate-11-1792505389180433400
- Ordinary 11-round replay: artifacts/repair-v15-plain-replay-1792505456007135800
- Archived round-9 boundary: artifacts/repair-v15-commit-audit-aggregate-11-1792505518342658800
- Original affected cell: results/repair-v15/v15-evaluation-fashion_mnist-aggregate-none-s11
- Reproduction script: scripts/audits/audit_repair_v15_commit.py

V16 retains the V15 algorithm and numerical protocol but adds independent CPU
arithmetic checks, validation-to-commit model binding, checks for mutated model
state, strict deterministic-operation mode, and a fresh process for each cell.
Fault-injection tests must confirm that these checks stop rather than silently
repair a corrupted model. All V16 cells, including calibration, are rerun and
sealed separately. A second integrity failure must stop the queue for investigation.
