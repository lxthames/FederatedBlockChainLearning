# V26 coordinate-median matrix: final integrity and analysis audit

Audit date: 2026-10-25 (Asia/Taipei). Source: frozen V26 coordinate-median
manifest and completed results under `artifacts/lifecycle-v26-coordinate` and
`results/lifecycle-v26-coordinate`. The read-only independent checker is
`scripts/audits/audit_lifecycle_v26_coordinate.py`.

## Integrity verdict

**Pass for completeness and file/trajectory integrity.** The checker found
200/200 unique manifest cells, 200/200 completed and sealed result directories,
0 failed cells, 20,000/20,000 passing model-integrity events, 400 repair events
on the expected rounds, and 400 matching offline observations. It verified the
frozen source snapshot, data, protocol, launcher, test file, per-output SHA-256
seals, summary identity fields, admission counts, and committed model hashes.
The 40 five-seed groups and 30 paired comparisons are complete; reported means
match the per-seed records. Run with:

`python scripts/audits/audit_lifecycle_v26_coordinate.py`

The coordinator's `publication_ready: false` is a frozen/default field that its
code never updates. It is **not** an integrity failure or an audit verdict.
This audit does not certify the package as publication-ready.

## Result interpretation

The pattern is conditional, not a universal defense claim. Periodic repair
lowered mean endpoint attack success rate (ASR) for the patch trigger in both
datasets under both attacker-stop and continuous attack. Under attacker-stop,
FEMNIST changed from 0.861 (quorum) to 0.047 (periodic), and Fashion-MNIST
from 0.886 to 0.086. Under continuous attack, the endpoint changed from 0.999
to 0.073 on FEMNIST and from 0.999 to 0.126 on Fashion-MNIST. However,
continuous-attack mean post-onset ASR remained high: 0.946 on FEMNIST and
0.736 on Fashion-MNIST. An endpoint-only account would hide that exposure.

The checkerboard trigger changes the conclusion. In continuous attack, mean
endpoint ASR under periodic repair was 0.999 on FEMNIST and 1.000 on
Fashion-MNIST, essentially the same as quorum. Under attacker-stop, periodic
repair had 0.740 versus quorum's 0.601 on FEMNIST, and 0.809 versus 0.858 on
Fashion-MNIST. These cells do not support a general repair benefit.

NAD-style repair was admitted infrequently: 33 admissions across its 50
trajectories and 200 attempts. In FEMNIST patch cells it had zero admissions
and reproduced the quorum outcome. Its checkerboard continuous-attack endpoint
ASR was 1.000 in both datasets. This is evidence about the audited adapter and
admission policy, not a broad verdict on NAD.

Coordinate median strongly reduced checkerboard ASR on Fashion-MNIST
(continuous endpoint 0.759 versus 1.000 for quorum), but did not solve the
same setting on FEMNIST (0.997 versus 1.000). It also had a substantial
FEMNIST clean-accuracy cost: 0.694 versus 0.766 for quorum. Its role is a
client/update baseline with a measured utility tradeoff, not a repair baseline.

All reported values above are five-seed means from the frozen matrix.
Confidence intervals and paired per-seed contrasts are in `analysis.json`.
The manuscript should show both endpoint and trajectory ASR, preserve trigger-
and dataset-specific results, and state the clean-utility and admission costs.

## Scope limit

The audit establishes internal consistency, sealing, and aggregation. It does
not establish external reproducibility on another machine, validate the
scientific suitability of every baseline implementation, or turn the failed
FLTrust pilot into a comparison. The V26 additions are one NAD-style repair
adapter and coordinate-median aggregation on two datasets, two triggers, and
five seeds. `publication_ready` remains false pending manuscript integration,
methodological review, and any independent replication required for the claim.
