# System Architecture

## Overview

This document describes the architecture of the Blockchain-Based Decentralized Federated Learning Framework. The system combines federated learning, blockchain technology, and consensus mechanisms to create a secure, private, and decentralized machine learning platform.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Application Layer                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │         EnhancedMultiHeadDNN Model                        │   │
│  │  (Parallel branches → Feature merge → Multi-head output) │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Federated Learning Layer                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Client 1   │  │   Client 2   │  │   Client N   │         │
│  │ Local Train  │  │ Local Train  │  │ Local Train  │         │
│  │   + Update   │  │   + Update   │  │   + Update   │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
│         │                 │                 │                  │
│         └─────────────────┴─────────────────┘                  │
│                           │                                     │
│                           ▼                                     │
│              ┌────────────────────────┐                        │
│              │  Model Aggregation     │                        │
│              │  (Staleness-aware)     │                        │
│              └────────────────────────┘                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Security & Validation Layer                   │
│  ┌──────────────────┐      ┌──────────────────┐               │
│  │  Proof-of-       │      │  Proof-of-       │               │
│  │  Personhood      │      │  Contribution   │               │
│  │  (PoP)           │      │  (PoC)           │               │
│  │                  │      │                  │               │
│  │  - Registration  │      │  - Accuracy      │               │
│  │  - Trust Score   │      │  - Loss          │               │
│  │  - Verification  │      │  - Robustness    │               │
│  └────────┬─────────┘      └────────┬─────────┘               │
│           │                          │                          │
│           └──────────┬───────────────┘                          │
│                     │                                          │
│                     ▼                                          │
│         ┌───────────────────────────┐                          │
│         │  Committee Consensus      │                          │
│         │  - Validation             │                          │
│         │  - Voting                 │                          │
│         │  - Score Aggregation      │                          │
│         └───────────────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Blockchain Layer                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  SignedBlockchain                                         │   │
│  │  - Transaction signing (HMAC)                            │   │
│  │  - Block mining                                           │   │
│  │  - Chain validation                                       │   │
│  │  - Persistence (JSON)                                     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Identity Management                                      │   │
│  │  - Client identities                                       │   │
│  │  - Signature generation                                    │   │
│  │  - Signature verification                                  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Storage & Persistence Layer                   │
│  ┌──────────────────┐      ┌──────────────────┐               │
│  │  Chain           │      │  Model            │               │
│  │  Persistence     │      │  Versioning      │               │
│  │  (JSON files)    │      │  (State dicts)   │               │
│  └──────────────────┘      └──────────────────┘               │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Storage Optimizer                                        │   │
│  │  - Historical block pruning                               │   │
│  │  - Model cleanup                                          │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Model Layer

#### EnhancedMultiHeadDNN
- **Purpose**: Deep neural network for federated learning
- **Architecture**:
  - Two parallel branches (128-dim each)
  - Feature merging via averaging
  - Multi-head output (one per client)
  - Softmax normalization
- **Key Features**:
  - Depth ≤ 3 layers
  - Batch normalization
  - Dropout regularization
  - Kaiming weight initialization

### 2. Federated Learning Layer

#### FederatedClient
- **Purpose**: Represents a participating client
- **Responsibilities**:
  - Local model training
  - Update computation
  - Data management

#### EnhancedBlockchainFederatedLearning
- **Purpose**: Main orchestration class
- **Responsibilities**:
  - Client initialization
  - Federated round coordination
  - Update aggregation
  - Model versioning

### 3. Security & Validation Layer

#### ProofOfPersonhood (PoP)
- **Purpose**: Prevent Sybil attacks
- **Methods**:
  - Challenge-based registration
  - Web-of-trust endorsements
- **Features**:
  - Trust scoring
  - Participation gating
  - Identity verification

#### ProofOfContribution (PoC)
- **Purpose**: Evaluate update quality
- **Metrics**:
  - Accuracy improvement (40%)
  - Loss reduction (40%)
  - Robustness score (20%)
- **Features**:
  - Threshold filtering
  - Top-K ranking
  - Contribution history

#### EnhancedCommitteeConsensus
- **Purpose**: Validate and vote on updates
- **Election Methods**:
  - Random selection
  - Score-based
  - Multi-factor
- **Features**:
  - Committee rotation
  - Voting aggregation
  - Score recording

### 4. Blockchain Layer

#### SignedBlockchain
- **Purpose**: Immutable record of updates
- **Features**:
  - Transaction signing
  - Block mining
  - Chain validation
  - Committee votes

#### IdentityManager
- **Purpose**: Manage client identities
- **Features**:
  - HMAC-based signing
  - Signature verification
  - Public ID management

### 5. Storage & Persistence Layer

#### ChainPersistence
- **Purpose**: Persist blockchain to disk
- **Format**: JSON
- **Features**: Save/load chain state

#### ModelVersion
- **Purpose**: Track model versions
- **Features**: Version incrementing, state storage

#### StorageOptimizer
- **Purpose**: Optimize storage usage
- **Features**: Historical block pruning, model cleanup

## Data Flow

### Training Flow

1. **Initialization**
   ```
   Global Model → Clients
   PoP Registration → Trust Scores
   ```

2. **Local Training**
   ```
   Client Data → Local Model → Updates
   ```

3. **Validation Pipeline**
   ```
   Updates → PoP Check → PoC Evaluation → Committee Vote → Blockchain
   ```

4. **Aggregation**
   ```
   Validated Updates → Weighted Average → Global Model Update
   ```

5. **Persistence**
   ```
   Global Model → Version Increment → Storage
   Blockchain → JSON Persistence
   ```

### Update Validation Flow

```
┌─────────────┐
│   Update    │
└──────┬──────┘
       │
       ▼
┌─────────────────┐      No      ┌──────────┐
│  PoP Verified?  │─────────────▶│ Reject   │
└──────┬──────────┘              └──────────┘
       │ Yes
       ▼
┌─────────────────┐      Low     ┌──────────┐
│  PoC Evaluation │─────────────▶│ Reject   │
└──────┬──────────┘              └──────────┘
       │ Pass
       ▼
┌─────────────────┐      No      ┌──────────┐
│ Committee Vote  │─────────────▶│ Reject   │
└──────┬──────────┘              └──────────┘
       │ Yes
       ▼
┌─────────────────┐
│  Blockchain     │
│  (Record)       │
└─────────────────┘
```

## Security Architecture

### Identity Management
- **HMAC Signing**: Each transaction signed with client's secret key
- **Public IDs**: Verifiable client identifiers
- **Signature Verification**: Ensures transaction authenticity

### PoP Security
- **Challenge Verification**: Prevents automated registrations
- **Trust Scoring**: Tracks participant reliability
- **Participation Gating**: Only verified participants can contribute

### PoC Security
- **Multi-Metric Evaluation**: Prevents gaming single metric
- **Threshold Filtering**: Rejects low-quality updates
- **Ranking System**: Prioritizes best contributions

### Blockchain Security
- **Immutable Records**: Once added, cannot be modified
- **Chain Validation**: Ensures integrity
- **Committee Votes**: Multiple validators required

## Scalability Considerations

### Current Limitations
- Single-process simulation
- JSON-based persistence (not suitable for large-scale)
- In-memory blockchain (limited by RAM)

### Future Enhancements
- Distributed P2P networking
- Database-backed persistence
- Sharded blockchain
- Light client support

## Performance Characteristics

### Training
- **Local Training**: O(epochs × batch_size × data_size)
- **PoC Evaluation**: O(validation_size × model_size)
- **Aggregation**: O(num_clients × model_size)

### Blockchain
- **Block Mining**: O(num_transactions)
- **Chain Validation**: O(chain_length)
- **Persistence**: O(chain_size)

## Error Handling

### Client Failures
- Staleness handling for missing updates
- Contribution gap tracking
- Automatic model synchronization

### Validation Failures
- PoP rejection: Client cannot participate
- PoC rejection: Update discarded
- Committee rejection: Update not recorded

### Blockchain Failures
- Chain validation on load
- Transaction rollback on errors
- Persistence error recovery

## Extension Points

### Custom Models
- Replace `EnhancedMultiHeadDNN` with any PyTorch model
- Implement custom forward pass
- Adjust aggregation logic

### Custom Consensus
- Implement new election methods
- Add custom validation rules
- Extend voting mechanisms

### Custom Storage
- Replace JSON persistence with database
- Implement IPFS integration
- Add distributed storage

## Related Documentation

- [System Design](SYSTEM_DESIGN.md): Detailed design decisions
- [Data Flow](DATA_FLOW.md): Complete data flow diagrams
- [API Reference](../api/API_REFERENCE.md): Component APIs

