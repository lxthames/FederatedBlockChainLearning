# Data Flow Documentation

## Overview

This document describes the complete data flow through the Blockchain-Based Decentralized Federated Learning system, from initialization to model updates.

## System Initialization Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    System Initialization                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────┐
        │  1. Load Training Data               │
        │     - SIFT features (X)             │
        │     - Client distributions (Y)        │
        └──────────────┬───────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────────┐
        │  2. Split Data                       │
        │     - Train/Validation split         │
        │     - Client data distribution       │
        └──────────────┬───────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────────┐
        │  3. Initialize Global Model          │
        │     - EnhancedMultiHeadDNN           │
        │     - Random weights                 │
        └──────────────┬───────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────────┐
        │  4. Initialize Blockchain            │
        │     - Genesis block                  │
        │     - Committee selection            │
        └──────────────┬───────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────────┐
        │  5. Initialize Clients               │
        │     - PoP registration               │
        │     - Local model copy               │
        │     - Data assignment                │
        └──────────────┬───────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────────┐
        │  6. Ready for Training               │
        └──────────────────────────────────────┘
```

## Federated Round Flow

```
┌─────────────────────────────────────────────────────────────┐
│              Federated Round (Iteration)                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 1: Local Training                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                │
│  │ Client 1 │  │ Client 2 │  │ Client N │                │
│  │          │  │          │  │          │                │
│  │ 1. Get   │  │ 1. Get   │  │ 1. Get   │                │
│  │   Model  │  │   Model  │  │   Model  │                │
│  │          │  │          │  │          │                │
│  │ 2. Train │  │ 2. Train │  │ 2. Train │                │
│  │   Locally│  │   Locally│  │   Locally│                │
│  │          │  │          │  │          │                │
│  │ 3. Compute│ │ 3. Compute│ │ 3. Compute│                │
│  │   Update │  │   Update │  │   Update │                │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                │
│       │             │             │                       │
│       └─────────────┴─────────────┘                       │
│                   Updates                                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 2: PoP Verification                                 │
│                                                             │
│  For each update:                                           │
│    ┌──────────────────────┐                                │
│    │ Check if client is   │                                │
│    │ PoP verified        │                                │
│    └──────┬───────────────┘                                │
│           │                                                 │
│      ┌────┴────┐                                            │
│      │         │                                            │
│     Yes       No                                            │
│      │         │                                            │
│      ▼         ▼                                            │
│  Continue   Reject                                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 3: PoC Evaluation                                   │
│                                                             │
│  For each PoP-verified update:                              │
│                                                             │
│  1. Load local model (with update)                         │
│  2. Load global model                                      │
│  3. Evaluate on validation set:                            │
│     - Accuracy improvement                                 │
│     - Loss reduction                                       │
│     - Robustness score                                      │
│  4. Compute overall score                                 │
│  5. Check threshold                                        │
│                                                             │
│  ┌──────────────────────┐                                  │
│  │ Score >= threshold?  │                                  │
│  └──────┬───────────────┘                                  │
│         │                                                   │
│    ┌────┴────┐                                              │
│    │         │                                              │
│   Yes       No                                              │
│    │         │                                              │
│    ▼         ▼                                              │
│  Accept    Reject                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 4: Ranking                                          │
│                                                             │
│  1. Sort accepted updates by score                         │
│  2. Apply top-K filter (if configured)                    │
│  3. Select final updates                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 5: Committee Validation                             │
│                                                             │
│  For each PoC-accepted update:                              │
│                                                             │
│  ┌──────────────────────────────────────┐                 │
│  │ Committee Members (PoP verified)     │                 │
│  │                                      │                 │
│  │  Member 1: Review → Vote → Score    │                 │
│  │  Member 2: Review → Vote → Score    │                 │
│  │  Member 3: Review → Vote → Score    │                 │
│  │  ...                                 │                 │
│  └──────────────┬───────────────────────┘                 │
│                 │                                           │
│                 ▼                                           │
│      ┌──────────────────────┐                              │
│      │ Aggregate Votes      │                              │
│      │ - Majority accept?    │                              │
│      │ - Median score        │                              │
│      └──────┬───────────────┘                              │
│             │                                               │
│        ┌────┴────┐                                           │
│        │         │                                           │
│       Yes       No                                           │
│        │         │                                           │
│        ▼         ▼                                           │
│     Accept    Reject                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 6: Blockchain Recording                             │
│                                                             │
│  For each accepted update:                                  │
│                                                             │
│  1. Create transaction:                                      │
│     - Client ID                                             │
│     - Model update (serialized)                            │
│     - PoC metrics                                          │
│     - Combined score                                        │
│     - Metadata                                              │
│                                                             │
│  2. Sign transaction:                                        │
│     - Generate HMAC signature                              │
│     - Include public ID                                     │
│                                                             │
│  3. Add to pending transactions                            │
│                                                             │
│  4. Mine block:                                             │
│     - Group transactions                                    │
│     - Record committee votes                                │
│     - Link to previous block                               │
│     - Calculate hash                                        │
│                                                             │
│  5. Persist to disk (JSON)                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 7: Model Aggregation                                 │
│                                                             │
│  1. Collect all validated updates                          │
│                                                             │
│  2. Calculate weights:                                      │
│     - Base weight = data_size / total_data_size            │
│     - Staleness penalty (if async)                         │
│     - Contribution gap adjustment                           │
│                                                             │
│  3. Weighted aggregation:                                   │
│     global_model = Σ(weight_i × update_i)                 │
│                                                             │
│  4. Update global model                                     │
│                                                             │
│  5. Increment version                                       │
│                                                             │
│  6. Store model state                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 8: Reward Distribution                              │
│                                                             │
│  For each accepted update:                                  │
│                                                             │
│  1. Calculate reward:                                       │
│     reward = base_reward × combined_score × contribution │
│                                                             │
│  2. Distribute to client                                    │
│                                                             │
│  3. Update contribution history                             │
│                                                             │
│  4. Track total rewards                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 9: Next Round Preparation                           │
│                                                             │
│  1. Distribute new global model to all clients              │
│                                                             │
│  2. Re-elect committee:                                     │
│     - Based on election method                             │
│     - Update committee members                             │
│                                                             │
│  3. Update client model versions                           │
│                                                             │
│  4. Cleanup old blocks (if needed)                         │
│                                                             │
│  5. Start next round                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Data Structures Flow

### Model Update Flow

```
Local Model State
    │
    ├─→ Compute Update (local - global)
    │
    ├─→ Serialize Update
    │   └─→ Shape information
    │   └─→ Norm values
    │
    ├─→ PoC Metrics
    │   ├─→ Accuracy improvement
    │   ├─→ Loss reduction
    │   └─→ Robustness score
    │
    ├─→ Transaction
    │   ├─→ Client ID
    │   ├─→ Serialized update
    │   ├─→ PoC metrics
    │   ├─→ Combined score
    │   ├─→ Signature
    │   └─→ Metadata
    │
    └─→ Blockchain Block
        ├─→ Transactions
        ├─→ Committee votes
        ├─→ Previous hash
        └─→ Block hash
```

### Model State Flow

```
Global Model
    │
    ├─→ Copy to Clients
    │   └─→ Local training
    │
    ├─→ Version Management
    │   ├─→ Current version
    │   ├─→ Version history
    │   └─→ State storage
    │
    └─→ Aggregation
        ├─→ Collect updates
        ├─→ Calculate weights
        └─→ Weighted average
```

## Error Flow

```
Error Occurred
    │
    ├─→ Client Error
    │   ├─→ Training failure
    │   │   └─→ Skip client this round
    │   │
    │   └─→ Update computation failure
    │       └─→ Use previous update or skip
    │
    ├─→ Validation Error
    │   ├─→ PoP verification failure
    │   │   └─→ Reject update
    │   │
    │   ├─→ PoC evaluation failure
    │   │   └─→ Reject update
    │   │
    │   └─→ Committee validation failure
    │       └─→ Reject update
    │
    └─→ Blockchain Error
        ├─→ Persistence failure
        │   └─→ Log error, continue
        │
        └─→ Chain validation failure
            └─→ Rollback or repair
```

## Performance Flow

### Optimization Points

1. **Parallel Processing**
   - Client training: Parallel across clients
   - PoC evaluation: Parallel across updates
   - Committee voting: Parallel across members

2. **Caching**
   - Model states: Cache recent versions
   - Validation results: Cache PoC scores
   - Blockchain: Cache recent blocks

3. **Batch Processing**
   - Group transactions: Batch into blocks
   - Aggregate updates: Process together
   - Persist: Batch writes

## Related Documentation

- [Architecture Overview](ARCHITECTURE.md)
- [System Design](SYSTEM_DESIGN.md)
- [API Reference](../api/API_REFERENCE.md)

