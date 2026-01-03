"""
Quick Reference: Second Layer Security Integration

Copy-paste code snippets for integrating 2nd layer security.
"""

# ============================================================================
# SNIPPET 1: Add Constants
# ============================================================================
# Add to your constants section:

# Second Layer Security Constants
STAKE_REQUIRED = 10.0  # Minimum stake to be committee candidate
PENALTY_AMOUNT = 5.0  # Amount slashed for approving poisoned model
ACCURACY_DROP_THRESHOLD = 0.05  # 5% drop triggers rollback
HISTORY_WINDOW = 5  # Rounds for moving average
ROLLBACK_DEPTH = 1  # Blocks to revert
INITIAL_STAKE_PER_CLIENT = 20.0  # Starting stake balance


# ============================================================================
# SNIPPET 2: Import Components
# ============================================================================
# Add to imports:

from scripts.second_layer_security import (
    StakingSystem, PoisoningDetector, RollbackBlockchain, SecondLayerSecurity
)


# ============================================================================
# SNIPPET 3: Update FederatedClient.__init__
# ============================================================================
# Add initial_stake parameter and stake_balance attribute:

def __init__(self, client_id: int, model: nn.Module, local_data: Tuple, 
             device: torch.device, val_split_ratio: float = 0.2,
             initial_stake: float = 0.0):  # ADD THIS
    # ... existing code ...
    self.stake_balance = initial_stake  # ADD THIS


# ============================================================================
# SNIPPET 4: Initialize Second Layer in EnhancedBlockchainFederatedLearning.__init__
# ============================================================================
# Add after existing initialization:

# Initialize second layer security
self.staking_system = StakingSystem(
    stake_required=STAKE_REQUIRED,
    penalty_amount=PENALTY_AMOUNT
)

# Initialize stake balances for all clients
for i in range(num_clients):
    self.staking_system.initialize_client(i, INITIAL_STAKE_PER_CLIENT)

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


# ============================================================================
# SNIPPET 5: Add Poisoning Check Hook
# ============================================================================
# In federated_round_async(), after metrics_logger.log_global_performance():

# After global performance logging
if metrics_logger is not None and metrics_logger.test_accuracy_history:
    current_accuracy = metrics_logger.test_accuracy_history[-1]
    
    # Check for poisoning (2nd Layer Security)
    if self.second_layer is not None:
        poisoning_detected, model_state = self.second_layer.check_and_respond(
            current_accuracy,
            self.global_model
        )
        
        if poisoning_detected and model_state is not None:
            # Revert model
            print("   Reverting global model to safe state...")
            self.global_model.load_state_dict(model_state)
            
            # Update model version
            current_block = self.rollback_blockchain.get_latest_block_index()
            self.model_version.current_version = current_block
            
            # Reset clients to reverted model
            for client in self.clients:
                client.apply_global_update(model_state)
                self.client_model_versions[client.client_id] = current_block
            
            print("   ✓ Model reverted successfully")


# ============================================================================
# SNIPPET 6: Checkpoint Model State
# ============================================================================
# After block mining, checkpoint the model:

if self.rollback_blockchain is not None:
    current_block = self.rollback_blockchain.get_latest_block_index()
    self.rollback_blockchain.checkpoint_model(
        current_block,
        self.global_model.state_dict()
    )


# ============================================================================
# SNIPPET 7: Update Committee Election with Stake Filtering
# ============================================================================
# Create new method or modify existing:

def elect_committee_with_staking(self) -> List[int]:
    """Elect committee with stake filtering"""
    # Get candidates (existing logic)
    sorted_nodes = sorted(
        self.node_scores.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    candidate_pool = [node_id for node_id, _ in sorted_nodes]
    
    # Filter by stake (2nd layer)
    if self.second_layer is not None:
        candidate_pool = self.second_layer.filter_committee_candidates(candidate_pool)
    
    committee = candidate_pool[:self.committee_size]
    
    # Lock stakes
    if self.second_layer is not None:
        self.second_layer.lock_committee_stakes(committee)
    
    return sorted(committee)


# ============================================================================
# SNIPPET 8: Update start_new_round()
# ============================================================================
# Modify to unlock stakes and use stake-filtered election:

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


# ============================================================================
# SNIPPET 9: Update initialize_clients()
# ============================================================================
# Add initial_stake parameter:

def initialize_clients(self, client_data_list: List[Tuple], initial_stake: float = 20.0):
    for i, (X_train, Y_train) in enumerate(client_data_list):
        # ... existing code ...
        client = FederatedClient(
            i, client_model, (X_train, Y_train), self.device,
            initial_stake=initial_stake  # ADD THIS
        )
        # ... rest of code ...


# ============================================================================
# SNIPPET 10: Print Security Summary After Training
# ============================================================================
# In main_blockchain_federated(), after training:

if fl_system.second_layer is not None:
    security_summary = fl_system.second_layer.get_security_summary()
    print("\n" + "="*70)
    print("SECOND LAYER SECURITY SUMMARY")
    print("="*70)
    print(f"Poisoning Events Detected: {security_summary['total_poisoning_events']}")
    print(f"Rollbacks Executed: {security_summary['total_rollbacks']}")
    print(f"Total Slashings: {security_summary['total_slashings']}")
    
    if security_summary['total_slashings'] > 0:
        print("\nSlashing Events:")
        for event in fl_system.staking_system.slashing_history:
            print(f"  Client {event['client_id']}: -{event['amount']:.2f} "
                  f"({event['reason']})")

