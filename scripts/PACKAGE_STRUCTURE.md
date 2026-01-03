# Complete Package Structure

## Overview

This package follows software engineering best practices with proper modularization, separation of concerns, and clean architecture.

## Directory Structure

```
scripts/
├── __init__.py                 # Main package exports
├── run.py                      # Main entry point
│
├── core/                       # Core components
│   ├── __init__.py
│   ├── models.py              # Neural network models
│   ├── blockchain.py         # Block, Blockchain, SignedBlockchain
│   ├── consensus.py          # Committee consensus mechanisms
│   ├── proof_of_personhood.py # PoP implementation
│   ├── proof_of_contribution.py # PoC implementation
│   ├── client.py              # FederatedClient class
│   ├── federated_learning.py  # Main orchestration class
│   ├── identity.py           # Identity management and signing
│   ├── persistence.py         # Blockchain persistence
│   └── utilities.py           # Versioning, staleness, incentives, storage
│
├── utils/                      # Utility functions
│   ├── __init__.py
│   ├── data_utils.py          # Data loading, distribution, evaluation
│   └── threshold.py            # Adaptive threshold functions
│
├── security/                   # Second layer security
│   ├── __init__.py
│   └── second_layer_security.py # Staking, poisoning detection, rollback
│
└── metrics/                    # Metrics and logging
    ├── __init__.py
    └── metrics_logger.py      # Metrics collection and visualization
```

## Module Responsibilities

### Core Modules

1. **models.py**: Neural network architecture
   - `EnhancedMultiHeadDNN`: Multi-head DNN with parallel branches

2. **blockchain.py**: Blockchain implementation
   - `Block`: Individual block structure
   - `Blockchain`: Basic blockchain
   - `SignedBlockchain`: Blockchain with signatures and persistence

3. **consensus.py**: Committee consensus
   - `CommitteeConsensus`: Basic consensus
   - `EnhancedCommitteeConsensus`: Enhanced with scoring, decay, term limits

4. **proof_of_personhood.py**: PoP mechanism
   - `ProofOfPersonhood`: Sybil attack prevention

5. **proof_of_contribution.py**: PoC mechanism
   - `ProofOfContribution`: Update quality evaluation

6. **client.py**: Client implementation
   - `FederatedClient`: Individual federated client

7. **federated_learning.py**: Main orchestration
   - `EnhancedBlockchainFederatedLearning`: Complete FL system

8. **identity.py**: Identity management
   - `IdentityManager`: Client identities and signatures

9. **persistence.py**: Persistence layer
   - `ChainPersistence`: Save/load blockchain state

10. **utilities.py**: Core utilities
    - `ModelVersion`: Version tracking
    - `ClientContributionTracker`: Contribution tracking
    - `StalenessHandler`: Staleness management
    - `IncentiveMechanism`: Reward distribution
    - `StorageOptimizer`: Storage management

### Utility Modules

1. **data_utils.py**: Data handling
   - `load_test_query_data()`: Load test queries
   - `setup_test_data_for_metrics()`: Prepare test data
   - `distribute_data_to_clients()`: Data distribution
   - `evaluate_global_model()`: Model evaluation

2. **threshold.py**: Threshold utilities
   - `get_adaptive_poc_threshold()`: Adaptive threshold calculation

### Security Modules

1. **second_layer_security.py**: Second layer security
   - `StakingSystem`: Staking and slashing
   - `PoisoningDetector`: Poisoning detection
   - `RollbackBlockchain`: Rollback capability
   - `SecondLayerSecurity`: Security orchestration

### Metrics Modules

1. **metrics_logger.py**: Metrics collection
   - `MetricsLogger`: Complete metrics tracking and visualization

## Usage

### Import from package:

```python
from core import EnhancedBlockchainFederatedLearning
from utils import get_adaptive_poc_threshold, distribute_data_to_clients
from metrics import MetricsLogger
from security import SecondLayerSecurity
```

### Run experiments:

```bash
python scripts/run.py
```

## Design Principles

1. **Separation of Concerns**: Each module has a single responsibility
2. **Modularity**: Components can be used independently
3. **Extensibility**: Easy to add new features
4. **Testability**: Each module can be tested independently
5. **Documentation**: All classes and functions are documented
6. **Type Hints**: Full type annotations for better IDE support

## Dependencies

- **Core**: torch, numpy, sklearn
- **Blockchain**: hashlib, json, time
- **Security**: collections (defaultdict)
- **Metrics**: matplotlib, json
- **Utils**: os, torch.utils.data

## Testing

Each module can be tested independently:

```python
from core.models import EnhancedMultiHeadDNN
from core.blockchain import Blockchain
# ... test individual components
```

## Extension Points

1. **New Models**: Add to `core/models.py`
2. **New Consensus**: Extend `core/consensus.py`
3. **New Metrics**: Extend `metrics/metrics_logger.py`
4. **New Security**: Extend `security/second_layer_security.py`

