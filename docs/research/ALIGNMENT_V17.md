# V17: qualification of aligned behavioral aggregate validation

Prepared 21 October 2026, Asia/Taipei, before execution. This is exposed-seed
development on seeds 11/17/23. No publication-readiness, priority, formal privacy,
Byzantine-validator, calibrated false-positive or broad backdoor-security claim.
V13-V16 sources, policies and results remain immutable. New code is isolated in
scripts/trust_alignment_v17; output in results/alignment-v17 and artifacts/alignment-v17.

## Finite sequence and prerequisites

The coordinator waits for the existing V16-fixed worker to release its OS lock
and complete all 114 cells without failures or holds. It verifies every V16
completion seal, source/data freeze, clean calibration binding, clean-clipping
calibration trajectory, method pairing and clean/attack pre-onset pairing.
A failed prerequisite stops the new queue; it does not restart V16 or relabel a failure.

Freeze all new sources, tests, this policy, the novelty assessment, launcher,
manifest, parent freeze and data bindings. Twelve passive reruns precede any
development: 2 datasets x 3 seeds x clean/targeted current-quorum conditions,
all 100 rounds. All six clean cells precede targeted reruns. Settings, data,
initialization, splits, offered participants, training and per-update validation
match V16 current exactly; experiment ID and protocol version are declared changes.
Every one of the 101 model hashes must match its sealed V16 reference. Extra
inference uses deterministic separate streams and never changes decisions.
Parity failure, corruption or crash stops the queue and preserves evidence.

## Scoring and telemetry

Retain sample-share-scaled clean-loss per-update quorum, fixed trust, clipping
ratio 2 and sample-weighted renormalization. Inspect the actual admitted aggregate.
No extra per-update veto, reputation, temporal score, MAD fit or omission repair.

Use the V16 deterministic two disjoint panels of five noncontributors and at most
eight private examples per validator. Require two examples and five responses
for each panel's score; insufficient response holds active gates and confers no
misconduct finding. Use the same 149-block inference bank as V16
(clean, 147 shared spatial transformations, and noise). All four new scoring
variants use the 147 shared spatial transformations only; independent Gaussian
noise is excluded from the factorial comparison. The original-gate control
retains its native noise score and original V16 clipping calibration.

For representation P (softmax probabilities) or H (one-hot argmax labels),
C_v(w,T,c)=mean_x[representation(w,Tx,c)-representation(w,x,c)].
Absolute response is C_v(candidate); incremental response is
C_v(candidate)-C_v(parent), preserving signed coordinate changes before pooling.
For each response tensor compute independent=maxima per validator then the
second-largest of five, or aligned=second-largest across validators at each
transformation/class then maximum across coordinates. Clamp the final scalar at
zero. Second-largest corresponds to two negative votes in a four-of-five quorum.
These label frequencies are observable behavior, not verified transformed labels.

The four fixed variants are probability_independent, probability_aligned,
label_independent and label_aligned. Also measure per-validator paired clean
CE excess=max(0,mean_change-2*SE). Store transformation IDs/specifications,
sample counts, hard class counts, both representations' paired contrast matrices,
panel membership, responses, scalar scores, clean metrics and scored-model hashes.
No raw image pooling or formal privacy claim; class counts/contrasts may leak information.
Engine decisions never read attacker identity, configured attack target, test arrays,
ASR or malicious-update acceptance. Those appear only in evaluation after decisions.

## Calibration and fixed diagnostic gate

After six clean passive completions, freeze leave-one-seed-out calibration:
for dataset/held-out seed/panel stage/variant, use other two seeds' 200 clean
panel scores for absolute and incremental higher empirical 99.5th percentiles.
Use their 1,000 validator clean-excess values for the clean threshold. A panel
passes both behavioral thresholds and requires four of five clean-excess passes.
A prospective round veto occurs if either panel fails. Exposed seeds and
correlated rounds give no independent false-positive guarantee. Freeze these
thresholds before reading targeted diagnostic outcomes.

After all 12 parity-verified reruns, assess label_aligned versus
probability_independent. Each of all six clean dataset/seed trajectories must
have <=3 prospective vetoes. For each FEMNIST seed, the V16 FedAvg targeted
control must have mean post-onset ASR >= matched clean trigger rate +10 points;
the current-quorum trajectory must contain at least one post-onset ASR>50%
round. On those rounds the label_aligned veto recall must improve by >=20 points
on every seed. Ineffective attacks, empty positive sets, missing responses and
incomplete data fail closed, without favorable-seed omission. Report every
variant and every seed, first alarm and high-ASR recall. This is a diagnostic
on fixed trajectories, not the accuracy/ASR of a retrained defense.

If this fixed gate fails, mark diagnostic_rejected, block all 72 development
cells, retain the negative results, and stop. No automatic threshold search or
alternative variant promotion. The sealed targeted primary-source novelty audit
supports only this bounded mechanistic test; it does not establish originality.

## Conditional end-to-end development

Only after the diagnostic passes and the frozen novelty assessment supports
bounded development: run 72 cells = 2 datasets x 3 seeds x clean/low_and_slow/
model_replacement x clipping/current/original/candidate, 100 rounds each.
All 24 clean cells first. There is no additional held-attack clean gate in this
matrix; failures stay visible as negative evidence. Settings and attacks inherit
V16 exactly. Clipping/current add no challenge. Original preserves per-update
quorum then the original V16 behavioral panel rule using its frozen native
thresholds. Candidate preserves the quorum then the label_aligned rule using
the held-out-seed clean fold. Both panels must pass; otherwise hold the round.
An already empty quorum aggregate is counted unsupported once, not twice.

Per seed targets: candidate clean and gradual utility no more than 2 points
below current; clean unsupported rounds <=3; zero monetary penalties. FEMNIST
mean post-onset and endpoint ASR must each be <=20%, each improve >=20 points
against current, and preserve targeted utility within 2 points of current to
exclude collapse. Matched clipping controls must establish targeted attack
effectiveness. Development gate failures do not trigger another search.

Report every seed and only complete matched group means; mean and endpoint
ASR, clean trigger rates, ordinary/lower-decile utility, honest exclusions,
accepted malicious sample weight, unsupported rounds, validation examples and
passes, communication estimates, runtime and artifact storage. Native clipping
still incurs inherited keep-all ranking evaluation work; disclose it. Runtime
includes integrity instrumentation. Passive diagnostic work is reported separately.

CPU-reference delta application, mutation checks during inference and validation,
and equality of scored/committed models remain active. Every completed cell
seals all files, including telemetry, integrity and passive parity reports.
Subprocess isolation, exclusive coordinator/cell locks, atomic status writes,
verified resume and preserved interrupted directories prevent silent result reuse.

## Execution and limits

Run tests/test_alignment_v17.py and tests/test_repair_v16.py, then:

```powershell
.venv-trust/Scripts/python.exe -m scripts.trust_alignment_v17.study --freeze
.\scripts\run_alignment_v17.ps1
```

The hidden background coordinator waits for V16 before acquiring GPU work.
Status, current cell, V16 audit, clean calibration, diagnostic decision and
conditional comparison live under artifacts/alignment-v17. The maximum is
84 new cells; no fresh-seed confirmation is automatically launched.

Passing these targets permits a separately frozen confirmation design with
exposure-audited fresh seeds, held-out evaluation data, unseen trigger families,
a defense-aware attacker and fidelity-audited FedVal/FLAME/CrowdGuard baselines.
New RNG seeds do not erase prior test-data exposure. Publication and acceptance
remain contingent on generalization, novelty and complete supporting evidence.
