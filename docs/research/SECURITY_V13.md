# V13: clean-first wide-CNN security development pilot

Prepared 20 October 2026, following the completed 60-cell V12 diagnostic.
Sources: `scripts/trust_security_v13/`. Outputs: `results/security-v13/`.
Coordinator artifacts: `artifacts/security-v13/`. Earlier versions stay frozen.

## Fixed candidate and scope

V12 wider-CNN FEMNIST means were FedAvg 64.37%, clipping 63.91%, renormalized
quorum 61.65%, and nominal-mass quorum 47.02%. Renormalized quorum passed all
three FEMNIST clean cells. Four-step training failed even with FedAvg at the
40-round budget. This selects a development candidate; it does not demonstrate
security or independently confirm utility.

Fix the candidate to `wide_cnn`, one local epoch, fixed trust and sample-weighted
renormalization of accepted clipped updates. No recovery, sanctions, EVM,
reputation tuning, tolerance grid, or architecture search is part of V13.
All data and seeds (11/17/23) remain development-exposed. Do not consult or launch
untouched-seed confirmation automatically.

## Finite matrix and order

Maximum **72 runs**: 2 datasets x 3 seeds x 4 methods x 3 conditions.

- Datasets: Fashion-MNIST and natural-writer 62-class FEMNIST, existing prepared
  training-only development holdouts and fixed evaluation groups.
- Methods: FedAvg; sample-weighted clipping-only; audited paper-based FedVal;
  proposed `scaled_renormalized`.
- Conditions: clean; bounded reversed-update `low_and_slow`; targeted
  `model_replacement`.

Execute all **24 clean cells first**, starting with Fashion-MNIST, then FEMNIST.
Then execute 24 gradual-attack and 24 targeted-attack cells. The candidate's
12 attack cells require every one of its six clean comparisons to pass.
If the gate fails, preserve explicit holds: 60 executions plus 12 held cells.
Controls execute their attack cells regardless of their own clean screen,
with failure labels. Weak clean baselines cannot establish superiority by
themselves. A missing or failed clean cell fails the candidate gate.

This study reruns the FEMNIST clean controls in V13. V12 evidence is preserved
and not silently pooled across source/protocol versions.

## Model, optimization and validation

All methods use identical wider CNNs: 3x3 padded convolutions with 32 then 64
channels, ReLU and max pooling after each, flatten, 128-unit ReLU hidden layer,
and task-class logits. No BatchNorm, dropout or output softmax. Parameter counts
are 428,350 for FEMNIST and 421,642 for Fashion-MNIST. Within each seed all
methods have identical initialization, splits and offered participant schedules.

Fifty clients, 40% participation, 30% potential poisoning identities with the
existing bounded sampler (six potential attackers among 20 participants).
Validator corruption is zero. Potential attacker roles affect the controlled
sampling schedule even in clean runs; they never filter honest proposals.
One local epoch with SGD momentum .9; optimizer resets each client-round.
Fashion-MNIST: alpha .5, learning rate .01, batch size 64, 30 rounds.
FEMNIST: natural writer splits, learning rate .05, batch size 16, 40 rounds.
No parameter adjustment after observing outcomes.

Candidate: clipping bound twice the round median update norm; evaluate each
proposal at its offered sample fraction on a proposer's private committee.
Honest validator approval requires paired mean loss change <=2 standard errors,
zero additive tolerance, and at least two local validation examples. Smaller
validation sets abstain. Require ceil(2C/3) approvals. Aggregate accepted clipped
updates with sample weights normalized to one; accept none means a zero update.
All trust channels are fixed and all trust rewards/rejection penalties are zero.
The 2-SE rule is a repeated-data heuristic, not a calibrated confidence or
Byzantine-security guarantee. Private validation bias remains a measured risk.

Clipping-only accepts all finite clipped updates. Its inherited keep-all ranking
path still incurs private evaluation work; report that overhead. FedAvg does not
clip. FedVal retains its native weighting and central class-complete reserved
reference examples, ten per class, rather than imposing private-committee access
or candidate clipping. Its adapter interpretation and limitations are fixed in
`TRUST_EXTENSION_BASELINE_AUDIT.md`, whose digest is included in the freeze.
This is not an independent reproduction of published FedVal accuracy.

## Attacks and effective-attack checks

Both attacks start at round 11, using the unchanged existing implementations:

- `low_and_slow`: multiply each attacker's own clean-trained update by
  `1 - 2.5 * min(attack_round / 10, 1)`, reaching -1.5 after ten active rounds.
  The bound is relative to that attacker's own update, not a measured honest
  reference norm. Although the simulator supplies peer updates in its context,
  this implementation does not use them. The inherited config field named
  `gradual_max_norm_ratio` supplies this maximum scalar magnitude. Label the
  measured threat precisely rather than inferring an honest-relative bound.
- `model_replacement`: existing local targeted poisoning (30% of malicious
  local examples, target class 0), followed by scaling 20/6 in the full pilot.
  This participant-count scaling does not guarantee exact model replacement
  under unequal sample weights.
  Preserve white corner trigger for Fashion and black corner trigger for the
  inverse-background FEMNIST. This is not an adaptive validation-aware attacker.

Do not strengthen attacks in place after observing V13. An ineffective attack is
an explicit limitation and cannot establish a successful defense.

Per seed, require matched FedAvg clean/attack data, training, sample schedules,
initialization and identical pre-onset model hashes (genesis through round 10).
Only the attack type may change in the clean/attack configuration pair.
Gradual poisoning is effective if it reduces FedAvg final accuracy by >=1 point.
Targeted replacement is effective if FedAvg mean post-onset ASR is >=10 points
above its matched clean post-onset trigger prediction rate. Endpoint ASR is
reported separately. These are fixed development screening bounds.

## Gates and interpretation

The candidate clean gate requires all six matched cells: FedAvg accuracy >=20%,
candidate accuracy loss <=5 points, <=3 unsupported rounds and zero honest
monetary penalty. No tolerance relaxation, seed selection or per-dataset attack
promotion after a partial gate.

Report all seed results and complete-three-seed means, final/round utility,
mean post-onset and endpoint ASR, clean trigger rate, lower-tail writer utility,
honest-update exclusion, malicious-update acceptance, unsupported rounds,
training steps/examples, private/reference evaluation counts, communication and
runtime. Honest exclusion covers all rounds for nonpoisoning clients; clean
exclusion includes all clients. Reused examples/rounds are not independent
statistical replicates.

Compare the candidate separately against FedAvg, clipping and the FedVal adapter.
A descriptive benefit screen is interpretable only when all three attacks are
effective against FedAvg, the candidate passes all six clean cells, each matched
reference passes its dataset's three clean cells, and pre-onset checks pass.
Require no candidate seed's attacked utility >5 points below the reference, no
honest monetary penalty, and >=1-point mean utility gain for gradual poisoning
or >=5-point mean post-onset ASR reduction for targeted poisoning. Beating only
FedAvg is insufficient evidence of advantage over stronger controls. These
screens neither establish significance nor automatically promote a method.

If clean and effective-attack results support the candidate against strong
controls, freeze a separate untouched-seed confirmation protocol. If they do
not, preserve the negative evidence and reconsider the defense claim. V13 does
not launch a confirmatory matrix in either case.

## Integrity and execution

Freeze all package sources, this protocol, the baseline audit, prepared data and
provenance, the complete manifest, and the V12 freeze digest. Verify V12's source,
data, policy and all 60 completion seals before starting. Require its coordinator
lock to be released. Execute V13 under an exclusive lock on CUDA, with hidden
background launch and stdout/stderr logs.

Seal config, summary, splits, events, chain trace and diagnostic inputs. V13
requires empty external aggregation inputs. Immutable completed summaries are
annotated in memory for analysis; no completed result is rewritten. On resume,
verify seals/configuration; preserve interrupted folders inside the result root.
Any source, data or policy change requires a new version. Missing/failed/held
cells prevent complete means and valid benefit claims.

```powershell
.venv-trust/Scripts/python.exe -W error::RuntimeWarning -m pytest tests/test_security_v13.py tests/test_budget_v12.py tests/test_screen_v11.py -q
.venv-trust/Scripts/python.exe -m scripts.trust_security_v13.study --freeze
.venv-trust/Scripts/python.exe -u -m scripts.trust_security_v13.study --run
```
