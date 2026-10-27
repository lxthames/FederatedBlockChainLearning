# Completed development evidence and next decision — 19 October 2026

Source: `artifacts/literature-v6-pilot/{analysis.json,comparison.md,status.json}`.
The original source and baseline-audit documents remain frozen. This companion
records outcomes without changing the protocol after observing them.

The V6 pilot completed 12 training cells, with no execution failures. Six attack
cells were held because their corresponding clean candidates exceeded the
five-point utility-loss gate. There were six completed conditional recovery
branches and three unsupported gradual branches with no detected suspects.

| Fashion-MNIST seed 11 | FedAvg | Rank/trust |
|---|---:|---:|
| Clean final utility | 73.2% | 75.9% |
| Gradual final utility | 11.1% | 68.7% |
| Targeted mean post-onset ASR | 67.0% | 30.5% |

Rank/trust and loss-quorum results were identical; recovery added no observed
benefit. V5 likewise completed 15 cells with zero recoveries and identical
outcomes across its three variants. Positive rank/trust effects support another
development replication; they do not validate recovery or confirm generality.

FLAME, FLShield cluster, and the partial FLDetector adaptation lost respectively
10.7, 5.5 and 6.8 clean utility points relative to FedAvg in V6. These results
apply to the disclosed implementation/settings, not a general failure of the
published methods. Correction from the completed V7 audit: the detector falsely
quarantined clean client 0 at round 22, despite zero measured honest-update
exclusion in its summary. It used coordinate median; the V7 matched median-only
control reached 66.33% versus 66.43% for the detector. This supports aggregation
as an explanation for most of the clean utility gap to FedAvg, while preserving
the false-quarantine finding. FLShield numerical warnings and validation-budget
differences remain relevant limitations.

For targeted recovery, the archive detector identified 2 of 15 malicious clients
and no honest clients. Thirteen attackers remained active during exact requests.
The attacked source endpoint ASR was 88.3% (distinct from its 67.0% temporal mean).
Detector-selected retraining ended at 88.4% ASR; the FedRecover adaptation at
66.9%. Oracle identities reduced endpoint ASR to 2.2% and 7.9% respectively.
Oracle Crab adaptation reached 0.1% ASR but only 52.8% utility. Oracle results are
conditional upper-bound diagnostics; no end-to-end detector/recovery claim follows.
The detector-selected Crab branch selected the latest checkpoint and performed
zero corrective requests. It supplies no evidence of useful recovery.

The principal unresolved recovery issue is reliable attacker attribution and
identifying a useful historical checkpoint. Committee attestations and local
EVM receipts do not establish poisoning or honest-validator culpability.
Monetary poisoning penalties remain disabled in the follow-up.

## Authorized next bounded study

See `RANK_REPLICATION_V7.md`: 12 matched rank/FedAvg replications on development
seeds 17/23 plus one clean median control. Seed 11 outputs are reused. Gates,
source/data hashes, and analysis are fixed before execution. No automatic
confirmatory launch follows. Final protocol selection, cross-dataset evaluation,
independent validator corruption, full seed groups, and manuscript reconciliation
remain outstanding.

## Separate original-workflow interruption

`artifacts/trust-2027/status.json` reports 379 completed cells and eight failed
cells; the workflow stopped in `core_confirmation` when the artifact-generation
subprocess failed at 2026-10-19 01:54 UTC. This status is not successful completion
of the full planned study. Existing results must be preserved and the artifact
error diagnosed before any resume; the V7 pilot does not restart that workflow.
