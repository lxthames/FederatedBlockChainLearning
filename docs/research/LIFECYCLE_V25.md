# V25 lifecycle evaluation protocol

Fixed before V25 outcome inspection, 22 October 2026. 400 full 100-round
trajectories: two datasets x five seeds x five conditions x eight arms.

Datasets: 62-class TFF FEMNIST, 50 natural writer clients; Fashion-MNIST, 50
Dirichlet-alpha-0.5 clients. Seeds: 4103, 4211, 4327, 4441, 4549, audited against
historical result summaries. FEMNIST excludes all enrolled and reference writers
recorded in historical data/**/natural-splits.json. New writer and reference
sets are also disjoint across the five seeds. Official per-writer train/test
correspondence is preserved; training/reference pixels matching evaluation
pixels are purged. This provides writer-level held-out evaluation relative to
the recorded experiments. It is not an independent external replication team.
Fashion-MNIST uses fresh partitions and seeds but its official images have
already been exposed; it is explicitly a replication dataset, not new test data.

Conditions: clean once per arm/seed/dataset; attacker-stop (poison rounds 11-50);
continuous poisoning (11-100). Each attacked condition is evaluated with both
the historical 2x2 corner patch and a centered 6x6 black/white checkerboard.
The checkerboard is outside the repair bank's uniform 2x2/4x4 transformations.
Attack: model replacement, target label 0, 30% malicious clients, bounded
participation. Model, optimizer and quorum settings inherit the V24/V17 lineage.
Serialized protocol_version describes that inherited runner; the V25 manifest
and source/data hashes define this new experiment.

Arms: quorum, clipping; V20 periodic repair, clean fine-tuning, Fine-Pruning
adaptation, each under private admission and under forced admission. Forced
admission is a diagnostic intervention, not a deployable defense. Repairs at
rounds 25/50/75/100. Providers and validators are assumed honest. All repair
methods use the same five providers, 32 examples per provider maximum, two
passes, 20 steps/provider/pass, batch size 16, learning rate .01, and disjoint
screen/verification panels. Optimizer steps and training example presentations
are matched. V20's extra teacher/proposal inference and pruning calibration are
measured separately; equal total FLOPs or elapsed time is not claimed. The
optional V22 peak-memory monitor is omitted because it does not affect periodic
admission; overhead comparisons therefore describe this shared V25 controller.

Fine-Pruning reference: Liu, Dolan-Gavitt and Garg, RAID 2018,
https://arxiv.org/abs/1805.12185 and
https://github.com/kangliucn/Fine-pruning-defense . Our explicit adaptation to
the shared small CNN ranks the 128 final hidden ReLU units by mean clean
repair-side activation. It tries cumulative pruning in 5% increments up to
95%, stopping at the first drop exceeding 2 percentage points on repair-side
clean accuracy. It then fine-tunes using the common budget. Zeroing incoming
weights, biases and outgoing weights leaves pruned ReLUs dead under subsequent
SGD; the model architecture/dimensions stay unchanged for FL aggregation.
This is a documented architectural/budget adaptation, not a claim to reproduce
the original face/speech experiments. Clean fine-tuning is a control, not a new
literature algorithm. No test data or verifier examples select pruning levels.
Private computation is simulated centrally; aggregate activation statistics and
pruning accuracy votes add disclosures and communication beyond V20.

Primary metric: seed-level mean committed ASR over rounds 11-100. Secondary:
post-stop mean ASR over rounds 51-100, endpoint ASR, final clean accuracy,
lower-decile client utility, repair admission rate, measured work and bytes.
Record source/candidate/committed clean accuracy and trigger rate after each
admission decision; these offline measurements cannot affect decisions.
Only complete five-seed groups receive bootstrap 95% confidence intervals.
Preserve all adverse outcomes; do not change thresholds after observing them.
Each trajectory must pass 100 model-integrity checks and is sealed on completion.

The manifest and transitive source/data hashes are frozen before full training.
The background launcher gives this lifecycle study GPU priority, preserving the
older TRUST workflow's completed cells and interrupting only its active cell.
A supervisor resumes the older workflow after the lifecycle process exits.
The coordinator verifies that no TRUST GPU worker remains before it trains,
then runs cells sequentially with independent worker processes. Failures are logged
and remaining cells continue; failed cells may be retried with the same freeze.
Current progress is artifacts/lifecycle-v25/status.json; each active cell has
progress.json. Results: results/lifecycle-v25. No automatic publication-ready
claim follows completion. Adaptive attackers, corrupted repair providers, and
repair-frequency sensitivity remain outside this finite protocol.
