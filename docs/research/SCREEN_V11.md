# V11 bounded clean-first development pilot

Prepared 20 October 2026 after the completed V10 results. New sources live in
`scripts/trust_screen_v11/`, output in `results/screen-v11/`, and coordinator
artifacts in `artifacts/screen-v11/`. V6-V10 data, source and results are preserved.
This is a follow-up development study; all seeds and development pools have been
exposed to prior analyses. It is not independent confirmation or evidence of novelty.

## Fixed hypotheses and methods

V10 showed that removing trust feedback did not eliminate FEMNIST clean harm.
Relative ranking can reject useful honest updates and accept harmful updates
because its cutoff is set by peer scores. V11 compares two prespecified changes:

1. `rank_no_penalty`: original private full-step rank voting and clipping ratio 2,
   reward .02, dynamic weights/eligibility, but zero rejection trust penalty.
   This isolates removing the rejection penalty; reward feedback remains.
2. `scaled_quorum`: no rank cutoff, fixed trust/weights/eligibility, clipping
   ratio 2, evaluate proposal i as w + a_i delta_i, a_i=n_i/sum_j n_j.
   On each validator's fixed private examples, let d be candidate loss minus
   baseline loss. Approve if mean(d) <= 2*sd(d)/sqrt(n), with zero additive
   tolerance and at least two examples. Fewer examples means abstention, not
   a negative vote. Form proposer-excluding committees with the existing rules
   and require ceil(2C/3) approvals. Apply sum_i accepted a_i delta_i: rejected
   mass contributes zero rather than renormalizing the survivors.

The two-standard-error rule is the same fixed backoff used in V9's margin probe,
now without mandatory rank selection. Repeated correlated validation and small
local class support prevent interpreting it as a calibrated confidence guarantee.
It can accept harmful low-signal updates; attacks must measure this. Local/global
objective mismatch persists. There is one rule and no tolerance grid or tuning
after observing results. Scaled voting and preserving nominal mass are a joint
candidate design here, not separately identified causal contributions.

Controls: FedAvg, sample-weighted clipping-only with all finite proposals passing,
and audited paper-based FedVal. Also rerun original rank/trust as a clean
diagnostic. Prior held original-rank attacks stay held; no original-rank attack
cells are in this manifest. Historical V10 results are not silently reused.
No recovery, stake slashing or EVM evidence is claimed.

FedVal audit: `TRUST_EXTENSION_BASELINE_AUDIT.md`, pinned author revision and
paper-based source interpretation. FedVal uses class-complete reserved reference
data and searches five weighting choices. Private validation and FedVal do not
have equal information or compute. Report reference-example counts, private and
reference example evaluations/forward passes, and run time. These counts capture
candidate validation work, not total FLOPs or all inference/training costs.

## Finite matrix and clean gates

Datasets Fashion-MNIST and natural-writer FEMNIST; development seeds 11/17/23.
50 clients, 40% participation, bounded 30% attacker roles, one local epoch,
SGD momentum .9, attack onset 11. Fashion-MNIST: 30 rounds, alpha .5, batch 64,
learning rate .01. FEMNIST: 40 rounds, batch 16, learning rate .05. Preserve
prepared training-only development holdouts and fixed evaluation groups.
Poisoning identities are distinct from validator corruption, which is zero here.

All 36 clean cells execute first: 2 datasets x 3 seeds x 6 methods.
Each candidate proceeds to its 12 attack cells only if all six clean cells pass:
matched initialization, data, training/client partitions, evaluation groups,
participant schedules and source; FedAvg clean utility >=20%; candidate utility
loss <=5 percentage points; unsupported rounds <=3; zero honest monetary penalty.
Missing/failed clean cells hold every attack cell for that candidate. No gate
relaxation or per-seed cherry-picking. These are inherited screening bounds, not
an equivalence, fairness or security certificate.

Attack controls always execute to diagnose attack effectiveness and baseline
limitations, with the clean gate displayed even when it fails. Two conditions:
the existing bounded reversed-update gradual attack and targeted model replacement.
3 controls x 2 datasets x 3 seeds x 2 conditions = 36 attack-control cells.
2 candidates x 2 datasets x 3 seeds x 2 conditions = at most 24 candidate attacks.
Maximum total **96** cells; complete failed-candidate holds reduce it to 72.

Attack validity: at least one percentage point final utility damage to FedAvg
for gradual poisoning, or targeted mean post-onset ASR at least ten points above
the matched clean trigger rate. Ineffective attacks cannot support a defense win.
Report mean post-onset ASR separately from endpoint ASR. Report all seeds, failed
and held cells, clean lower-tail utility, benign-update exclusion, voting reasons,
runtime and validation work. Group means require all three matched seeds.

Assess benefit only after effective attacks and clean preservation. Inherited
development screening targets: at least one-point mean gradual utility gain or
five-point targeted ASR reduction versus FedAvg, no seed's attacked utility more
than five points below FedAvg, and no monetary penalty. Beating FedAvg alone
does not establish advantage over clipping/FedVal, significance or novelty.
The coordinator makes no automatic method promotion or confirmatory launch.

## Integrity and execution

Freeze full package sources, this protocol, baseline audit and data/provenance
hashes before running. An exclusive process lock prevents duplicate coordinators.
Completed artifacts require file-digest seals; interrupted output is preserved.
Analysis annotates in memory and never rewrites completed summaries. A changed
source/data/policy requires a new version. CUDA is required for the study;
tests use small independent CPU configurations. Native reference budgets remain
explicit and are not manipulated to manufacture a comparison win.

```powershell
.venv-trust/Scripts/python.exe -m pytest tests/test_screen_v11.py -q
.venv-trust/Scripts/python.exe -m scripts.trust_screen_v11.study --freeze
.venv-trust/Scripts/python.exe -u -m scripts.trust_screen_v11.study --run
```

Background launches use a hidden window with stdout/stderr logs. Stop at this
finite pilot and review the evidence before selecting a full experiment.
