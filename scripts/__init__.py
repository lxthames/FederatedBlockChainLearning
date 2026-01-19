"""
Federated Blockchain Learning Package

A complete implementation of blockchain-based federated learning with:
- Proof-of-Personhood (PoP) for Sybil resistance
- Proof-of-Contribution (PoC) for update quality assessment
- Committee consensus for decentralized validation
- Asynchronous federated learning with staleness handling
- Metrics logging and visualization
- Second layer security (staking, rollback, poisoning detection)
"""

__version__ = "1.0.0"
__author__ = "Federated Learning Team"

# Core imports
try:
    from .core.models import EnhancedMultiHeadDNN
    from .core.blockchain import Block, Blockchain, SignedBlockchain
    from .core.consensus import CommitteeConsensus, EnhancedCommitteeConsensus
    from .core.proof_of_personhood import ProofOfPersonhood
    from .core.proof_of_contribution import ProofOfContribution
    from .core.client import FederatedClient
    from .core.federated_learning import EnhancedBlockchainFederatedLearning
except ImportError:
    # Fallback if core modules not found
    pass

# Utilities
try:
    from .utils.data_utils import (
        load_test_query_data,
        load_test_query_with_ground_truth,
        setup_test_data_for_metrics,
        distribute_data_to_clients,
        evaluate_global_model
    )
    from .utils.threshold import get_adaptive_poc_threshold
except ImportError:
    pass

# Security - try both locations
try:
    from .security.second_layer_security import (
        StakingSystem,
        PoisoningDetector,
        RollbackBlockchain,
        SecondLayerSecurity
    )
except ImportError:
    try:
        from .second_layer_security import (
            StakingSystem,
            PoisoningDetector,
            RollbackBlockchain,
            SecondLayerSecurity
        )
    except ImportError:
        pass

# Metrics - try both locations
try:
    from .metrics.metrics_logger import MetricsLogger
except ImportError:
    try:
        from .metrics_logger import MetricsLogger
    except ImportError:
        pass

__all__ = [
    # Core
    'EnhancedMultiHeadDNN',
    'Block',
    'Blockchain',
    'SignedBlockchain',
    'CommitteeConsensus',
    'EnhancedCommitteeConsensus',
    'ProofOfPersonhood',
    'ProofOfContribution',
    'FederatedClient',
    'EnhancedBlockchainFederatedLearning',
    # Utils
    'load_test_query_data',
    'load_test_query_with_ground_truth',
    'setup_test_data_for_metrics',
    'distribute_data_to_clients',
    'evaluate_global_model',
    'get_adaptive_poc_threshold',
    # Security
    'StakingSystem',
    'PoisoningDetector',
    'RollbackBlockchain',
    'SecondLayerSecurity',
    # Metrics
    'MetricsLogger',
]

