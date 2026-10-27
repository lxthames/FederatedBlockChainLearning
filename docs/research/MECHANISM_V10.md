# V10 mechanism-study implementation

20 October 2026. Separately versioned implementation of the issues in
`VALIDATION_STRATEGY_AUDIT_2026_10_20.md`. Historical V6-V9 sources and results
remain immutable. V10 is a validation-only, memory-backed development study;
its contract/recovery behavior is not certified by this change.

## Implemented semantics

- Committee decisions carry response, negative and expected-member counts.
  Approval requires the prescribed quorum. A quality rejection requires enough
  received negative votes to make approval impossible even if every missing
  member approves. Otherwise the outcome is insufficient response. Missing
  responses do not themselves penalize the proposer. An explicit legacy switch
  exists solely to reproduce the previous accounting error.
- Rank scores remain relative. Logs separately report raw mean local loss change,
  local harm, rank rejection and decision basis. None establishes misconduct.
  The runner defaults to no trust penalty for rank rejection. The historical
  penalty policy must be selected explicitly for mechanism diagnosis; it is
  not promoted as a mitigation or proof of malicious intent.
- Trust updates, aggregation weighting and validator eligibility can be fixed
  independently. Fixed eligibility uses the baseline trust threshold but still
  requires sufficient stake. Monetary poisoning penalties and recovery are off.
- Evaluation groups are computed once from the original partition before the
  intervention. Natural FEMNIST writer groups remain fixed; Fashion-MNIST groups
  remain explicitly synthetic. Client IDs and sample counts are retained.
- Validation redistribution uniformly permutes the existing validation-only pool
  without replacement, preserving every per-validator count. It improves coverage
  only in expectation and cannot guarantee absent classes. It requires different
  data access and is a diagnostic control, not a private deployment solution.
- Pairing binds training partitions, original validation allocation, validation
  union/counts, reference allocation, fixed evaluation groups, initialization,
  training settings, full data/test arrays, source and participant schedule.
  Only declared private-versus-redistributed differences are permitted. Old
  summaries lacking these bindings fail closed.

## Bounded diagnostic matrix

42 clean cells: two datasets (Fashion-MNIST, FEMNIST), development seeds 11/17/23,
and seven arms:

1. Private validation, dynamic trust weights/eligibility/updates.
2. Redistributed validation, dynamic trust weights/eligibility/updates.
3. Private validation, fixed trust weights/eligibility/updates.
4. Redistributed validation, fixed trust weights/eligibility/updates.
5. Private validation, dynamic weights/updates but fixed eligibility.
6. FedAvg.
7. Clipping only, all proposals rank-accepted, fixed trust.

Dynamic diagnostic arms explicitly retain the historical rank-rejection penalty
to measure its effects. They use corrected missing-response handling. With honest
responsive validators, the private dynamic arm preserves original learning
semantics; parity is tested against V9. The default no-rank-penalty policy is an
implementation safety choice, not a claim that its utility or security is better.

Training follows V8 development settings: 50 clients, 40% participation, one
local epoch, onset parameter 11 (no attacks executed), 30% potential-attacker
identities only for matching the bounded participant schedule. All actual updates
are benign. Fashion-MNIST uses 30 rounds, alpha .5, SGD .01, momentum .9, batch
64; FEMNIST uses natural writer partitions, 40 rounds, SGD .05, momentum .9,
batch 16. The same prepared training-only development holdouts are used. These
seeds are search-exposed, not independent confirmation.

Primary descriptive outputs: paired clean utility, macro/worst/lower-decile
utility with fixed groups, nonacceptance reason counts, trust/eligibility
trajectories and unsupported rounds. Every clean diagnostic cell is retained,
including harmful configurations. There is no attack expansion, promotion gate
or automatic confirmation. The fifth arm isolates an eligibility intervention
under dynamic weighting; this is not a complete factorial of all channels.
Analysis reports per-seed trust, redistribution and eligibility contrasts plus
the validation-by-trust interaction. Three-seed means are emitted only for
complete matched groups; a missing or incompatible cell cannot silently enter
an interaction or group mean. These are descriptive development effects.

## Commands and artifact integrity

Run from the repository root:

```powershell
# Inspect only; does not train or write a frozen manifest.
.venv-trust/Scripts/python.exe -m scripts.trust_mechanism_v10.study
# Verify data provenance and freeze the source, policy, data and 42-cell manifest.
.venv-trust/Scripts/python.exe -m scripts.trust_mechanism_v10.study --freeze
# Explicit execution of the frozen finite GPU study.
.venv-trust/Scripts/python.exe -m scripts.trust_mechanism_v10.study --run
```

Study artifacts: `artifacts/mechanism-v10/`; results: `results/mechanism-v10/`.
An OS-backed coordinator lock prevents duplicate runs. Source/policy/data
changes stop reuse. Completed cells have file-digest seals; unsealed interrupted
folders are preserved before rerunning. Summaries are not rewritten during
analysis. No V9 result is silently reused or relabeled. Any background launch
must use a hidden window and redirected logs.

Tests: `pytest tests/test_mechanism_v10.py tests/test_bias_v9.py tests/test_comparison_v8.py -q`.
The tests exercise missing votes, original learning parity, fixed evaluation
groups, invalid pairing, trust-channel isolation and completion-file tampering.
Passing software tests does not establish robustness, fairness or novelty.

Implementation verification on 20 October: 32 tests passed, including 17 V10
cases and the existing V8/V9 regression suites. All 33 required data/provenance
files verified against the prepared development data. The V9 source fingerprint
still matches its frozen manifest. No 42-cell training batch was launched as
part of this implementation task.
