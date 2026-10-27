# Loss-monitor diagnostic pilot (19 October 2026)

Status: development only. No publication or confirmatory readiness claim.
V4 sources, results and deployed contract bytecode remain unchanged.

## Recorded failure mechanism

`artifacts/recovery-v4-diagnosis/report.json` binds each inspected event log by
SHA256 and records all seven benefit checks in the nine guarded-recovery runs.
Six checks occurred under attack and one on clean data. None passed the joint
accuracy-degradation/paired-benefit quorum. Across these checks the surviving
attestations were 0--7 against a quorum of 10. In addition, some accuracy alarms
failed the preceding local-accuracy quorum. Two targeted runs never reached a
benefit check. This does not establish that removing the accuracy condition will
work; it motivates a measured comparison.

## Candidate and mathematical interpretation

V5 retains rank voting, clipping, dynamic trust and the same paired loss margin.
At the end of each five-observation window (checkpoint plus four updates), each
fixed panel member compares current and checkpoint cross-entropy on exactly the
same private examples. Let d_j = CE(current,j) - CE(checkpoint,j). The member
approves restoration if mean(d) > 0.02 + 2 sd(d)/sqrt(n), n >= 2.
Restoration requires ceil(2C/3) approvals from the full fixed panel, with unique
members. Accuracy decline is logged but is no longer a prerequisite for this
candidate. There is one loss decision per completed window; no threshold search.
Restoration resets panel/history and preserves economic and append-only history.
All monetary recovery penalties are disabled in this pilot.

This is an empirical benefit heuristic, not a confidence sequence, poison proof,
or general backdoor detector. Reused private data and heterogeneous writers
invalidate a simple independent-sample significance interpretation. Persistent
harm present before a window can be invisible; a historical checkpoint may be
equally poisoned or less useful. Clean-loss invariance can hide targeted attacks.

Literature grounding: [Zeno](https://proceedings.mlr.press/v97/xie19b.html)
motivates validation-based suspicion/ranking; this is not an implementation of
its theorem or a transfer of its oracle assumptions. [FedRecover](https://arxiv.org/abs/2210.10936)
addresses recovery after malicious clients are detected, using historical updates
and additional training. A simple checkpoint restore is not FedRecover and does
not inherit its results. These sources support separating detection, restoration
benefit and continued poisoning; neither validates this proposed rule.

## Frozen, bounded pilot

Entry point: `.venv-trust/Scripts/python.exe -m scripts.trust_recovery_v5.study`.
Artifacts: `artifacts/loss-monitor-v5-pilot/`; results:
`results/loss-monitor-v5-pilot/`. One CUDA coordinator, exclusive process lock.

15 runs: three variants (rank/trust without recovery, V4 accuracy-plus-benefit
rule, direct paired-loss quorum) times five conditions (clean; gradual and
targeted attacks beginning at round 1; the same attacks beginning at round 21).
Development seed 11 only, FEMNIST training-only holdout, 50 clients, 40 rounds,
40% participation, 30% attackers, SGD .05/momentum .9, one epoch, batch 16.
Data, code, manifest and reused V4 contract artifact are fingerprinted before work.
The three clean runs precede attacks. A candidate must lose at most five utility
points against recovery disabled, have at most four unsupported rounds and no
honest monetary penalties. A failed clean candidate's attack runs are held.

Delayed attack is an additional threat setting, not a replacement for the failed
immediate attack. It tests the value of an earlier useful checkpoint; it supplies
no checkpoint identity or attack onset to the detector. Comparisons are paired
within each onset, and attacked-round ASR excludes warmup. Immediate attacks and
all unsuccessful results remain reported. A one-seed pilot cannot support seed
confidence intervals or general robustness claims. No automatic expansion.

Metrics include final and attacked-round utility, attacked-round ASR, all alarm
rounds, pre-onset alarms, restoration utility gain/regret and honest penalties.
Existing per-round FPR/FNR retain their limited attack-presence interpretation.
Full EVM trace replay is deferred until a candidate merits further evaluation;
the pilot is a detector diagnostic, not new deployment validation.

Next gate: inspect full pilot outcomes before considering a three-seed replication,
benign drift/threshold tests, independent corrupted validators, or cross-dataset
confirmation. If neither recovery rule helps, report the negative result and
reconsider the paper's recovery claim rather than extending runtime blindly.

## Automatic assessment

`scripts/data_preparation/assess_loss_monitor.py --watch` waits for a terminal
coordinator status, then writes `assessment.json` and `assessment.md` under the
pilot artifacts directory. It starts no experiments and changes no run artifacts.
The assessment policy was added after the pilot launched, during its first clean
run. It is an exploratory decision aid, not a preregistered statistical analysis.
It verifies frozen inputs, config hashes, complete round records, append-only
trace lengths, matched participant lists/data and identical pre-onset model
commitments. Missing/duplicate cells and failed/held executions block readiness.

An individual condition merits replication review only if clean recovery caused
no alarms, clean utility loss is at most .05, attacked utility loss at most .05,
there are no pre-onset alarms or honest monetary penalties, and recovery adds
at least .01 final utility for gradual poisoning or .05 attacked-round ASR
reduction for targeted poisoning. These margins are development screens. The
report keeps unsuccessful conditions, residual ASR and restoration regret.
No result of this one-seed assessment grants publication readiness.
