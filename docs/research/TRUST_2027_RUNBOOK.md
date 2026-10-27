# TRUST 2027 v2 implementation and experiment runbook

The v2 protocol replaces training-set evaluation in the original harness.
Legacy 4/8-client and synthetic outputs remain pilot/debugging artifacts.
The new manuscript is under `Details/ACM_TRUST_2027/`.

## Environment

`.venv-trust` is a separate Python environment inheriting the installed CUDA
PyTorch stack through system-site-packages. It is not a fully hermetic runtime.
Exact resolved packages and code hashes are recorded under `artifacts/trust-2027/`.
The observed GPU is an RTX 2080 Ti; PyTorch is CUDA 12.6-enabled.
For a fresh environment install a compatible CUDA PyTorch/torchvision pair,
then pytest, numpy, PyYAML, web3, matplotlib and PyMuPDF. Keep the frozen
environment report with the experiment artifacts.

```powershell
.\.venv-trust\Scripts\python.exe -m pytest -q
$env:TRUST_EVM_TESTS='1'
.\.venv-trust\Scripts\python.exe -m pytest tests/test_evm_integration.py -q
```

## Local blockchain

```powershell
cd blockchain
npm.cmd ci
npm.cmd run compile
npm.cmd run node
```

The node binds only to 127.0.0.1:8545, chain 31337, with 100 local test accounts.
The Python adapter refuses other chain IDs. These unlocked accounts are strictly
development identities. The registrar and off-chain orchestrator are trusted.
Each run deploys a separate contract and records its address and receipts.

```powershell
.\.venv-trust\Scripts\python.exe -m scripts.research.cli run --config configs/trust-live-evm.json --output results/new-live-run
.\.venv-trust\Scripts\python.exe -m scripts.research.evm --config results/new-live-run/config.json --trace results/new-live-run/chain-trace.json --output results/new-replay
```

Use a fresh output directory. `transactions.jsonl` stores actual receipt-backed
gas, calldata bytes and local latency. One stake credit is 10^9 wei. It has no
inferred dollar price. Model weights, private examples and checkpoints remain
off-chain. Candidate-vote authentication uses actual account transactions;
the in-memory hash field is only a diagnostic checksum, not a signature scheme.

## Background study

```powershell
.\.venv-trust\Scripts\python.exe -m scripts.research.workflow prepare
.\.venv-trust\Scripts\python.exe -m scripts.research.workflow run
```

The background launcher records its PID, stdout/stderr, and atomic `status.json`
under `artifacts/trust-2027/`. Development seeds 3/5/7 check attack effectiveness;
confirmatory seeds 13/29/47/71/101 are separate. The core has 300 cells. Large
studies may take days on one GPU; do not infer completion from a running process.
`attack-validation.json` records attack gates, and `held-experiments.json` names
confirmatory cells not run because their attack was ineffective or unvalidated.
`failures.jsonl` records exceptions and configuration information.
The threshold grid is a prespecified sensitivity study. Default settings are
not optimized against confirmatory outcomes.

Every run preserves config, data/test fingerprints, full split indices, JSONL
events, model-version trace and summary. Model checkpoints used by recovery are
currently in memory for that run; optimizer-level process resume is not supported.
An interrupted seed restarts fully into a new directory while retaining old logs.
Completed seeds are skipped only when the configuration hash matches.

## Scientific interpretation

- Median committee quality is not an unbiased estimate under non-IID data.
- Detector FPR/FNR score attack presence in a round, not causal proof of damage.
- Gradual poisoning is bounded sign reversal, not an adaptive stealth optimizer.
- ALIE receives honest reference updates: an explicit omniscient attack assumption.
- Multi-Krum uses a declared ceil(participants * fraction) Byzantine bound;
  rounds whose assumptions fail produce failed cells, without fallback.
- The TRUST core uses bounded participation: a floor(participants * fraction)
  poisoning-client quota sampled without replacement plus honest participants.
  This is a controlled attack schedule, not defense access to attacker labels.
  The uniform-sampling option instead fails explicitly if actual attackers exceed
  a baseline's declared budget. Do not mix the two sampling policies.
- BOBA follows the authors' two-stage affine-subspace algorithm. A correction
  reconstructs the final latent mean instead of averaging unprojected vectors.
- Runtime includes runner and, for live mode, transaction overhead; receipt
  latency is also reported separately. Communication is an explicit payload
  estimate, not a network capture, and excludes transport framing.
- Recovery gains and regrets are pre/post checkpoint differences; clean-matched
  trajectory comparisons are needed for longer-term causal interpretations.

## SIFT provenance requirement

Update, 19 October 2026: the author selected replacement labels. The verified
SIFT1M root is `data/sift1m-verified/`, and its dedicated background job is
`artifacts/sift1m-study/status.json`. The following paragraph describes the old
unverified arrays only; those arrays remain excluded. The natural-client/FedVal
follow-up has a separate [runbook](TRUST_EXTENSION_RUNBOOK.md) and output root.

The existing arrays contain 1,000,000 training rows with 128 dimensions and
4/8-shard targets. Shapes alone cannot validate normalization or row correspondence.
Case-study loading therefore requires `data/sift-provenance.json` with documented
`query_target_correspondence_verified: true`, `test_feature_divisor`, and source
provenance. Do not assert those fields without evidence. No larger-client labels
are synthesized. Once verified, the v2 SIFT matrix compares actual EMH-DNN and MLP
at the same top-1 budget. Until then this part of the paper stays pending.

## Build and inspect the manuscript

```powershell
.\.venv-trust\Scripts\python.exe Details/ACM_TRUST_2027/build.py
```

Complete seed groups alone enter tables/plots. Paired sign-flip tests operate on
five seed-level differences and use Holm correction; p-values are coarse with
only five seeds. Inspect every rendered page after rebuilding. Draft and evidence
status remain explicit until the final author audit.
