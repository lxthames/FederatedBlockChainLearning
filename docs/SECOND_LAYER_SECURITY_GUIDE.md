# Second Layer Security Implementation Guide

## Overview

This guide explains how to implement the **2nd Layer of Security** combining:
1. **Model Rollback (Time Machine)** - Reactive recovery from poisoning
2. **Staking & Slashing (Skin in the Game)** - Proactive economic security

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              First Layer (Existing)                         │
│  - PoP Verification                                         │
│  - PoC Evaluation                                           │
│  - Committee Consensus                                      │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│           Second Layer (NEW)                                │
│  ┌──────────────────────┐  ┌──────────────────────┐        │
│  │  Staking System      │  │  Poisoning Detector │        │
│  │  - Stake locking     │  │  - Accuracy monitor │        │
│  │  - Slashing          │  │  - Drop detection   │        │
│  └──────────┬───────────┘  └──────────┬───────────┘        │
│             │                          │                   │
│             └──────────┬───────────────┘                   │
│                        ▼                                   │
│              ┌──────────────────────┐                      │
│              │  Rollback Blockchain    │                      │
│              │  - Model checkpoints │                      │
│              │  - State reversion   │                      │
│              └──────────────────────┘                      │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. PoisoningDetector
- Monitors test accuracy over rounds
- Detects sudden drops (>5% threshold)
- Uses moving average for stability

### 2. StakingSystem
- Manages stake balances per client
- Locks stakes for committee members
- Slashes stakes for bad behavior
- Tracks slashing history

### 3. RollbackBlockchain
- Wraps existing blockchain
- Stores model checkpoints per block
- Enables reversion to previous states

### 4. SecondLayerSecurity
- Orchestrates all components
- Coordinates poisoning detection and response
- Manages stake filtering for elections

## Configuration Constants

Add these to your constants section:

```python
# Second Layer Security Constants
STAKE_REQUIRED = 10.0  # Minimum stake to be committee candidate
PENALTY_AMOUNT = 5.0  # Amount slashed for approving poisoned model
ACCURACY_DROP_THRESHOLD = 0.05  # 5% drop triggers rollback
HISTORY_WINDOW = 5  # Rounds for moving average
ROLLBACK_DEPTH = 1  # Blocks to revert (1 = previous block)
INITIAL_STAKE_PER_CLIENT = 20.0  # Starting stake balance
```

## Step-by-Step Implementation

### Step 1: Import Second Layer Components

Add to your imports:

```python
from scripts.second_layer_security import (
    StakingSystem, PoisoningDetector, RollbackBlockchain, SecondLayerSecurity
)
```

### Step 2: Update FederatedClient Class

Add `stake_balance` attribute:

```python
class FederatedClient:
    def __init__(self, client_id: int, model: nn.Module, local_data: Tuple, 
                 device: torch.device, val_split_ratio: float = 0.2,
                 initial_stake: float = 0.0):  # ADD THIS
        # ... existing code ...
        self.stake_balance = initial_stake  # ADD THIS
```

### Step 3: Initialize Second Layer in EnhancedBlockchainFederatedLearning

Add to `__init__`:

```python
def __init__(self, ..., enable_second_layer: bool = True,
             initial_stake_per_client: float = 20.0):
    # ... existing initialization ...
    
    # Initialize second layer security
    if enable_second_layer:
        self.staking_system = StakingSystem(
            stake_required=STAKE_REQUIRED,
            penalty_amount=PENALTY_AMOUNT
        )
        
        # Initialize stake balances
        for i in range(num_clients):
            self.staking_system.initialize_client(i, initial_stake_per_client)
        
        self.poisoning_detector = PoisoningDetector(
            accuracy_drop_threshold=ACCURACY_DROP_THRESHOLD,
            history_window=HISTORY_WINDOW
        )
        
        self.rollback_blockchain = RollbackBlockchain(self.blockchain)
        
        self.second_layer = SecondLayerSecurity(
            blockchain=self.rollback_blockchain,
            staking_system=self.staking_system,
            poisoning_detector=self.poisoning_detector,
            proof_of_personhood=self.proof_of_personhood,
            rollback_depth=ROLLBACK_DEPTH
        )
```

### Step 4: Update Committee Election with Stake Filtering

Modify committee election to filter by stake:

```python
def elect_committee_with_staking(self) -> List[int]:
    # Get candidates (existing logic)
    sorted_nodes = sorted(self.node_scores.items(), key=lambda x: x[1], reverse=True)
    candidate_pool = [node_id for node_id, _ in sorted_nodes]
    
    # Filter by stake (2nd layer)
    if self.second_layer is not None:
        candidate_pool = self.second_layer.filter_committee_candidates(candidate_pool)
    
    committee = candidate_pool[:self.committee_size]
    
    # Lock stakes
    if self.second_layer is not None:
        self.second_layer.lock_committee_stakes(committee)
    
    return sorted(committee)
```

### Step 5: Add Poisoning Check After Global Performance

In `federated_round_async()`, after `metrics_logger.log_global_performance()`:

```python
# After global performance logging
if metrics_logger is not None and metrics_logger.test_accuracy_history:
    current_accuracy = metrics_logger.test_accuracy_history[-1]
    
    # Check for poisoning
    if self.second_layer is not None:
        poisoning_detected, model_state = self.second_layer.check_and_respond(
            current_accuracy,
            self.global_model
        )
        
        if poisoning_detected and model_state is not None:
            # Revert model
            self.global_model.load_state_dict(model_state)
            # Update versions and reset clients
            # ... (see integration code)
```

### Step 6: Checkpoint Model State

After block mining, checkpoint the model:

```python
if self.rollback_blockchain is not None:
    current_block = self.rollback_blockchain.get_latest_block_index()
    self.rollback_blockchain.checkpoint_model(
        current_block,
        self.global_model.state_dict()
    )
```

### Step 7: Unlock Stakes After Round

In `start_new_round()`:

```python
def start_new_round(self):
    # Unlock previous committee stakes
    if self.second_layer is not None:
        previous_committee = self.committee.committee_members
        self.second_layer.unlock_committee_stakes(previous_committee)
    
    # Re-elect with stake filtering
    if self.second_layer is not None:
        self.committee.committee_members = self.elect_committee_with_staking()
    else:
        self.committee.start_new_round()
    
    self.blockchain.committee_members = self.committee.committee_members
```

## How It Works

### Staking Flow

1. **Initialization**: Each client starts with initial stake (e.g., 20 tokens)
2. **Election**: Only clients with stake ≥ STAKE_REQUIRED can be committee members
3. **Locking**: When elected, stake is locked in escrow
4. **Service**: Committee validates updates
5. **Unlocking**: After round, stake is unlocked (if no issues)
6. **Slashing**: If poisoning detected, stake is burned

### Rollback Flow

1. **Checkpointing**: After each block, model state is saved
2. **Monitoring**: PoisoningDetector monitors accuracy
3. **Detection**: If accuracy drops >5%, poisoning detected
4. **Response**:
   - Identify bad committee (who signed last block)
   - Slash their stakes
   - Ban them (trust_score = 0)
   - Revert blockchain to previous block
   - Restore model from checkpoint
5. **Recovery**: Training continues from safe state

## Example Scenario

### Round 10: Normal Operation
- Committee: [1, 3, 7] (all have sufficient stake)
- Stakes locked: 10 tokens each
- Model updates accepted
- Accuracy: 0.85

### Round 11: Poisoning Attack
- Committee: [2, 5, 8] (Sybil nodes)
- They approve poisoned updates
- Model accuracy drops to 0.75 (10% drop)
- **PoisoningDetector triggers!**

### Response (Automatic)
1. **Detect**: Drop = (0.85 - 0.75) / 0.85 = 11.8% > 5% threshold
2. **Identify**: Bad committee = [2, 5, 8]
3. **Slash**: Each loses 5 tokens (PENALTY_AMOUNT)
4. **Ban**: Trust scores set to 0
5. **Revert**: Blockchain rolled back to Round 10
6. **Restore**: Model restored from Round 10 checkpoint
7. **Continue**: Training resumes from safe state

## Benefits

### Defense in Depth
- **Staking**: Prevents attacks (economic cost)
- **Rollback**: Recovers from attacks (reactive)

### Economic Security
- Attackers must "pay" to attack (stake requirement)
- Bad behavior is punished (slashing)
- Honest behavior is rewarded (stake accumulation)

### Resilience
- System can recover from poisoning
- Model history is preserved
- Automatic detection and response

## Configuration Tuning

### Staking Parameters
- **STAKE_REQUIRED**: Higher = more security, fewer eligible members
- **PENALTY_AMOUNT**: Higher = stronger deterrent
- **INITIAL_STAKE**: Starting balance for clients

### Detection Parameters
- **ACCURACY_DROP_THRESHOLD**: Lower = more sensitive (may false positive)
- **HISTORY_WINDOW**: Larger = more stable, slower to detect
- **ROLLBACK_DEPTH**: How many blocks to revert

### Recommended Settings

**Conservative (High Security)**:
```python
STAKE_REQUIRED = 20.0
PENALTY_AMOUNT = 10.0
ACCURACY_DROP_THRESHOLD = 0.03  # 3% drop
HISTORY_WINDOW = 3
ROLLBACK_DEPTH = 1
```

**Balanced (Default)**:
```python
STAKE_REQUIRED = 10.0
PENALTY_AMOUNT = 5.0
ACCURACY_DROP_THRESHOLD = 0.05  # 5% drop
HISTORY_WINDOW = 5
ROLLBACK_DEPTH = 1
```

**Aggressive (Fast Detection)**:
```python
STAKE_REQUIRED = 5.0
PENALTY_AMOUNT = 2.0
ACCURACY_DROP_THRESHOLD = 0.10  # 10% drop
HISTORY_WINDOW = 10
ROLLBACK_DEPTH = 2
```

## Testing

### Test Case 1: Normal Operation
- Run training with honest clients
- Verify no false positives
- Check stakes are locked/unlocked correctly

### Test Case 2: Poisoning Attack
- Inject malicious updates
- Verify detection triggers
- Check rollback executes
- Verify slashing occurs

### Test Case 3: Stake Filtering
- Set some clients with low stake
- Verify they're excluded from committee
- Check only high-stake clients elected

## Monitoring

After training, check:

```python
if fl_system.second_layer is not None:
    security_summary = fl_system.second_layer.get_security_summary()
    print(f"Poisoning Events: {security_summary['total_poisoning_events']}")
    print(f"Rollbacks: {security_summary['total_rollbacks']}")
    print(f"Slashings: {security_summary['total_slashings']}")
```

## Files Created

1. **`scripts/second_layer_security.py`**: Complete implementation
2. **`scripts/second_layer_integration.py`**: Integration examples
3. **This guide**: Implementation instructions

## Next Steps

1. Import second layer components
2. Update FederatedClient with stake_balance
3. Initialize SecondLayerSecurity in EnhancedBlockchainFederatedLearning
4. Add poisoning check hook
5. Update committee election with stake filtering
6. Test with normal and attack scenarios

## Related Documentation

- [Global Model Protection](GLOBAL_MODEL_PROTECTION.md)
- [Architecture Documentation](ARCHITECTURE.md)
- [Metrics Implementation](METRICS_IMPLEMENTATION_GUIDE.md)

