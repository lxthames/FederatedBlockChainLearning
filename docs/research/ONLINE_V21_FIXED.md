# V21-fixed: historical audit compatibility

This implements exactly the experiment specified in
[ONLINE_V21.md](ONLINE_V21.md), with the audit correction documented in
[ONLINE_V21_FAILURE_AUDIT.md](ONLINE_V21_FAILURE_AUDIT.md).

The fixed freeze binds the original V21 freeze, failure audit, fixed sources,
tests and launcher, plus the existing V20 and historical references. Historical
sample-weight validation reconstructs the unavailable summary field from
recorded accepted IDs and local sample counts. All round/model/budget matching
requirements remain. No defense setting, scoring threshold, schedule, condition,
seed, clean gate or final project target is changed.

Artifacts: artifacts/online-v21-fixed. Cells: results/online-v21-fixed.
The original failed study is retained in artifacts/online-v21 and
results/online-v21. All 12 clean cells precede the 24 conditional attack cells.
The same no-tuning and no-automatic-confirmation stop rules apply.
