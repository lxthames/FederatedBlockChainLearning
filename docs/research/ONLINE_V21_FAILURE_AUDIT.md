# V21 historical-summary audit failure

The initial V21 first control completed 100 rounds on clean FEMNIST seed 11.
Its post-run baseline audit failed with KeyError:
accepted_malicious_weight_history. That field exists in the inherited V17/V21
summary but was never present in the historical V16 summary. V16 also lacks
mean_post_onset_malicious_weight and accepted_weight events.

This was a compatibility bug in the new audit, missed by toy tests that compared
model trajectories but did not run the historical-schema validation. The initial
V21 sources, freeze, failed status, stdout/stderr and unsealed first cell remain
unchanged. No V21 repair cell or attack condition had started.

Read-only inspection verified all 101 model hashes (genesis plus 100 rounds),
all accuracy values, partitions/participants, local work, private validation
work and client rejection counts match the historical control exactly. Its
final clean accuracy was 80.0252047889099%, identical to the reference. This
is a control result; it supplies no new online V20 efficacy evidence.

The fix uses a separately frozen V21-fixed coordinator and output roots. It
reconstructs malicious accepted sample mass from historical per-round accepted
client IDs, local_update_diagnostic sample_count and malicious-role metadata,
respecting attack onset. It verifies all rounds and accepted sample counts;
missing/duplicate observations fail closed. This is only offline audit code.
It supplies no identities or attack outcomes to repair or admission. If a
historical summary already has weight telemetry, that must match reconstruction.

The online runner, V19/V20 repair, coefficients, deployment, allocation, panels,
36-cell matrix, clean qualification and publication gates are unchanged. The
initial 100-round cell is preserved and independently audited; the fixed study
reruns controls rather than declaring the failed unsealed cell complete.
