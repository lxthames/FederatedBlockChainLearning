"""
Core modules for Federated Blockchain Learning

This package contains the core components:
- Models: Neural network architectures
- Blockchain: Blockchain implementation
- Consensus: Committee consensus mechanisms
- Proof mechanisms: PoP and PoC
- Client: Federated client implementation
- Federated Learning: Main orchestration class
"""

from .models import EnhancedMultiHeadDNN
from .blockchain import Block, Blockchain, SignedBlockchain
from .consensus import CommitteeConsensus, EnhancedCommitteeConsensus
from .proof_of_personhood import ProofOfPersonhood
from .proof_of_contribution import ProofOfContribution
from .client import FederatedClient
from .federated_learning import EnhancedBlockchainFederatedLearning

__all__ = [
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
]

