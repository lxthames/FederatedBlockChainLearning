# V26 baseline audit and launch

The V26 confirmation matrix contains 200 trajectories: two datasets, four methods,
five lifecycle settings, and five seeds. The methods are quorum, periodic V20
repair, NAD-style teacher/attention repair, and coordinate-median aggregation.
The settings are clean/patch, attacker-stop/patch, continuous/patch,
attacker-stop/checkerboard, and continuous/checkerboard.

The NAD adapter is frozen as `nad_v1`. At each repair round it trains a clean
teacher from the repair panels, then updates the poisoned parent with clean
cross-entropy plus normalized last-convolution attention transfer to the teacher.
It uses the same repair allocation, 200 student steps, 200 teacher steps, panel
separation, and admission wrapper as the V25 repair arms. Test arrays, attacker
labels, and trigger definitions are not passed to the adapter.

The client/update baseline is coordinate median from the audited aggregation
implementation. Its clean FEMNIST pilot completed with 100/100 integrity checks
and final accuracy 70.19%. This low utility is retained as a measured baseline.

A FLTrust pilot was attempted under the V25 configuration and failed its native
precondition (`no update has positive trust alignment`). The failed cell is kept
under `artifacts/lifecycle-v26`; no FLTrust result is included in the coordinate-
median matrix and no aliasing is used.

The coordinate-median matrix is frozen under
`artifacts/lifecycle-v26-coordinate/frozen-inputs.json` and launched with 200
cells. Completion and failure counts are written to
`artifacts/lifecycle-v26-coordinate/status.json`.
