# Federated Blockchain Learning

> **Research status:** The original 4/8-client outputs and legacy `scripts/run.py`
> are pilot artifacts, not publication evidence. New journal experiments must use
> the tested `scripts.research` harness and
> `docs/research/EXPERIMENT_PROTOCOL.md`.

## Legacy prototype package

The original notebook code is retained under `scripts/` for traceability. It is
superseded for new evidence by the journal stress-test harness described below.

## 📁 Package Structure

```
scripts/
├── __init__.py                    # Package exports
├── run.py                         # Main entry point
│
├── core/                          # Core components
│   ├── models.py                 # Neural network models
│   ├── blockchain.py             # Blockchain implementation
│   ├── consensus.py              # Committee consensus
│   ├── proof_of_personhood.py    # PoP mechanism
│   ├── proof_of_contribution.py  # PoC mechanism
│   ├── client.py                 # Federated client
│   ├── federated_learning.py     # Main orchestration
│   ├── identity.py               # Identity management
│   ├── persistence.py            # Blockchain persistence
│   └── utilities.py              # Versioning, staleness, etc.
│
├── utils/                         # Utility functions
│   ├── data_utils.py             # Data handling
│   └── threshold.py              # Adaptive threshold
│
├── security/                      # Second layer security
│   └── second_layer_security.py  # Staking, rollback, poisoning
│
└── metrics/                       # Metrics and logging
    └── metrics_logger.py          # Metrics collection
```

## 🚀 Quick Start

### Run Experiments

```bash
cd project
python scripts/run.py
```

This will run experiments with:
- Client scales: 4, 8
- Distribution: skewed
- Rounds: 20

### Configuration

Edit `scripts/run.py` to modify:
- `CLIENT_SCALES = [4, 8]`
- `DISTRIBUTIONS = ["skewed"]`
- `FEDERATED_ROUNDS = 20`

## 📦 What's Included

### ✅ All Features

1. **Core FL System**
   - EnhancedMultiHeadDNN model
   - Blockchain with signatures
   - Committee consensus
   - PoP verification
   - PoC evaluation
   - Asynchronous FL
   - Staleness handling
   - Contribution tracking
   - Incentive mechanism

2. **Enhanced Features**
   - Adaptive PoC threshold
   - Score decay in committee
   - Term limits
   - Metrics logging (3 hooks)
   - Test query integration
   - Second layer security

3. **Complete Integration**
   - All hooks in federated_round_async
   - Metrics logger integration
   - Adaptive threshold
   - Complete run.py

## 📝 Usage

### Import from Package

```python
from core import EnhancedBlockchainFederatedLearning
from utils import get_adaptive_poc_threshold
from metrics import MetricsLogger
```

### Run with All Features

The `run.py` script includes:
- ✅ Adaptive threshold
- ✅ Metrics logging
- ✅ Test query integration
- ✅ All security features

## 🔍 File Locations

- **Main script**: `scripts/run.py`
- **Core classes**: `scripts/core/`
- **Utilities**: `scripts/utils/`
- **Security**: `scripts/security/` (or `scripts/second_layer_security.py`)
- **Metrics**: `scripts/metrics/` (or `scripts/metrics_logger.py`)

## ✅ Verification

All code has been:
- ✅ Extracted from notebook
- ✅ Organized into modules
- ✅ Properly documented
- ✅ Type hinted
- ✅ Following software engineering principles

## Journal stress-test harness

```powershell
python -m pip install -r requirements-research.txt
python -m pytest -q
python -m scripts.research.cli run --config configs/smoke.yaml --output results/smoke
```

See `docs/research/REPRODUCIBILITY.md` for matrix export and dataset requirements.
This repository is a simulator and does not claim production blockchain,
formal privacy, or Proof-of-Personhood guarantees.

