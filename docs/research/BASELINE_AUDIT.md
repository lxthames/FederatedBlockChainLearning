# Baseline implementation audit for TRUST v2

The adapters operate on local SGD model deltas. Their original theoretical
assumptions need not hold for heterogeneous local multi-step training. This
audit verifies algorithmic operations; it is not a reproduction of each paper's
published accuracy results.

| Method | Operations and parameters | Primary reference |
|---|---|---|
| FedAvg | Training-sample-weighted mean of local deltas | https://proceedings.mlr.press/v54/mcmahan17a.html |
| Coordinate median | Per-coordinate tensor median; even-sized sets use the lower middle order statistic | https://proceedings.mlr.press/v80/yin18a.html |
| Multi-Krum | Squared Euclidean distances, n-f-2 nearest peers per score; select max(1,n-2f-2) minimum-score updates, then mean; require n >= 2f+3 | https://proceedings.neurips.cc/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html |
| FLTrust | ReLU cosine weights against reserved root delta; client deltas normalized to root norm; normalized trust-weighted mean | https://www.ndss-symposium.org/ndss-paper/fltrust-byzantine-robust-federated-learning-via-trust-bootstrapping/ |
| BOBA | Initialize c-1-dimensional affine subspace from class-wise root vectors; iteratively fit n-f nearest updates; compute simplex coefficients; p_min=0 with n-f fallback; reconstruct accepted latent mean | https://proceedings.mlr.press/v238/bao24a.html |

BOBA source cross-check: the paper's official software link resolves to
https://github.com/baowenxuan/BOBA . The implementation in
`src/defense/BOBA.py` aggregates in latent space and inverse-transforms the mean.
The earlier local adapter incorrectly averaged unprojected selected updates;
v2 corrects that operation and tests that off-subspace contamination cannot
re-enter the result through that averaging step. PyTorch SVD implements the
affine PCA stage; no upstream source code is vendored.

All methods reserve the same reference pool for comparable client partitions;
only FLTrust and BOBA access its examples. The proposed framework never uses
it to vote or monitor. All methods share the same held-out test data.

The primary matrix uses a declared bounded poisoning-client participation
schedule. The defense does not know which participants are malicious. The
Byzantine budget for Multi-Krum and BOBA is ceil(n * configured fraction),
not a value fitted from model performance. Infeasible bounds and insufficient
BOBA class-subspace support are recorded as failed cells without algorithm
substitution. Uniform participation is supported, with explicit failure when
sampled attackers exceed the declared budget.
