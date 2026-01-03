"""
Integration Guide: Second Layer Security (Rollback + Staking)

Shows how to integrate the 2nd layer security into the federated learning system.
"""

# ============================================================================
# Step 1: Update FederatedClient Class
# ============================================================================
# Add to FederatedClient.__init__:

class FederatedClient:
    def __init__(self, client_id: int, model: nn.Module, local_data: Tuple, 
                 device: torch.device, val_split_ratio: float = 0.2,
                 initial_stake: float = 0.0):  # ADD THIS PARAMETER
        self.client_id = client_id
        self.model = model
        self.device = device
        self.local_epochs = 0
        
        # ADD: Economic state
        self.stake_balance = initial_stake  # Initial stake balance
        
        # ... rest of existing code ...


# ============================================================================
# Step 2: Initialize Second Layer Security
# ============================================================================
# In EnhancedBlockchainFederatedLearning.__init__:

from scripts.second_layer_security import (
    StakingSystem, PoisoningDetector, RollbackBlockchain, SecondLayerSecurity
)

class EnhancedBlockchainFederatedLearning:
    def __init__(self, num_clients: int, committee_size: int, input_size: int,
                 num_outputs: int, device: torch.device,
                 election_method: str = "score", enable_async: bool = True,
                 persistence_path: str = "blockchain_state.json",
                 poc_threshold: float = 0.1,
                 top_k_updates: Optional[int] = None,
                 enable_second_layer: bool = True,  # ADD THIS
                 initial_stake_per_client: float = 20.0):  # ADD THIS
        # ... existing initialization ...
        
        # Initialize second layer security
        if enable_second_layer:
            self.staking_system = StakingSystem(
                stake_required=STAKE_REQUIRED,
                penalty_amount=PENALTY_AMOUNT
            )
            
            # Initialize stake balances for all clients
            for i in range(num_clients):
                self.staking_system.initialize_client(i, initial_stake_per_client)
            
            self.poisoning_detector = PoisoningDetector(
                accuracy_drop_threshold=ACCURACY_DROP_THRESHOLD,
                history_window=HISTORY_WINDOW
            )
            
            # Wrap blockchain with rollback capability
            self.rollback_blockchain = RollbackBlockchain(self.blockchain)
            
            self.second_layer = SecondLayerSecurity(
                blockchain=self.rollback_blockchain,
                staking_system=self.staking_system,
                poisoning_detector=self.poisoning_detector,
                proof_of_personhood=self.proof_of_personhood,
                rollback_depth=ROLLBACK_DEPTH
            )
        else:
            self.second_layer = None
            self.staking_system = None
            self.poisoning_detector = None
            self.rollback_blockchain = None


# ============================================================================
# Step 3: Update Committee Election (Filter by Stake)
# ============================================================================
# In EnhancedCommitteeConsensus.elect_committee() or EnhancedBlockchainFederatedLearning:

def elect_committee_with_staking(self, method: str = None) -> List[int]:
    """Elect committee with stake filtering"""
    method = method or self.election_method
    
    # Get candidate pool (existing election logic)
    if method == "score":
        sorted_nodes = sorted(
            self.node_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        candidate_pool = [node_id for node_id, _ in sorted_nodes]
    else:
        candidate_pool = list(range(self.total_clients))
    
    # Filter by stake requirements (2nd layer security)
    if self.second_layer is not None:
        candidate_pool = self.second_layer.filter_committee_candidates(candidate_pool)
    
    # Select top candidates
    committee = candidate_pool[:self.committee_size]
    
    # Lock stakes for elected members
    if self.second_layer is not None:
        self.second_layer.lock_committee_stakes(committee)
    
    return sorted(committee)


# ============================================================================
# Step 4: Add Poisoning Check After Global Performance Logging
# ============================================================================
# In federated_round_async(), after metrics_logger.log_global_performance():

def federated_round_async(self, local_epochs: int = 1, batch_size: int = 128,
                          lr: float = 0.001, max_staleness: int = 3,
                          metrics_logger=None) -> Dict:
    # ... existing code ...
    
    # After aggregation and global performance logging:
    if validated_updates:
        self.aggregate_updates_async(validated_updates)
        new_version = self.model_version.increment_version()
        self.model_version.store_version(new_version, self.global_model.state_dict())
        self.storage_optimizer.store_model(new_version, self.global_model.state_dict())
        self._update_contributions_and_rewards(validated_updates, validation_scores)
        
        # Log global performance
        if metrics_logger is not None:
            metrics_logger.log_global_performance(
                model=self.global_model,
                device=self.device
            )
            
            # ================================================================
            # HOOK: Check for Poisoning and Rollback (2nd Layer Security)
            # ================================================================
            if self.second_layer is not None and metrics_logger.test_accuracy_history:
                current_accuracy = metrics_logger.test_accuracy_history[-1]
                
                # Check for poisoning
                poisoning_detected, model_state = self.second_layer.check_and_respond(
                    current_accuracy,
                    self.global_model
                )
                
                if poisoning_detected:
                    # Rollback detected! Revert model
                    if model_state is not None:
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
                        print("   ⚠️ Training will continue from reverted state")
        
        # Checkpoint model state in blockchain
        if self.rollback_blockchain is not None:
            current_block = self.rollback_blockchain.get_latest_block_index()
            self.rollback_blockchain.checkpoint_model(
                current_block,
                self.global_model.state_dict()
            )


# ============================================================================
# Step 5: Unlock Stakes After Round
# ============================================================================
# In start_new_round() or after block mining:

def start_new_round(self):
    """Start new round and unlock previous committee stakes"""
    # Unlock stakes from previous committee
    if self.second_layer is not None:
        previous_committee = self.committee.committee_members
        self.second_layer.unlock_committee_stakes(previous_committee)
    
    # Re-elect committee (with stake filtering)
    if self.second_layer is not None:
        # Use stake-filtered election
        self.committee.committee_members = self.elect_committee_with_staking()
    else:
        self.committee.start_new_round()
    
    self.blockchain.committee_members = self.committee.committee_members


# ============================================================================
# Step 6: Update initialize_clients to Set Initial Stakes
# ============================================================================
# In initialize_clients():

def initialize_clients(self, client_data_list: List[Tuple], initial_stake: float = 20.0):
    """Initialize clients with initial stake"""
    self.clients = []
    total_data_size = 0
    
    for i, (X_train, Y_train) in enumerate(client_data_list):
        # Register with PoP
        success, token = self.proof_of_personhood.register_node(i, verification_method="challenge")
        if not success:
            continue
        
        client_model = copy.deepcopy(self.global_model)
        client = FederatedClient(
            i, client_model, (X_train, Y_train), self.device,
            initial_stake=initial_stake  # ADD THIS
        )
        self.clients.append(client)
        total_data_size += client.data_size
        
        # Initialize stake in staking system
        if self.staking_system is not None:
            self.staking_system.initialize_client(i, initial_stake)
        
        self.contribution_tracker.update_sample_share(i, client.data_size, total_data_size)
        print(f"✓ Client {i} registered with PoP (token: {token[:8]}...) [Stake: {initial_stake}]")


# ============================================================================
# Step 7: Reward Distribution (Add Stake Rewards)
# ============================================================================
# In _update_contributions_and_rewards():

def _update_contributions_and_rewards(self, validated_updates: List[Tuple], 
                                     validation_scores: Dict[int, float]):
    total_data_size = sum(update[0]["data_size"] for update in validated_updates)
    client_rewards = {}
    
    for update_info, metrics, score in validated_updates:
        cid = update_info["client_id"]
        contrib = (update_info["data_size"] / total_data_size) * score
        self.contribution_tracker.record_contribution(cid, contrib)
        
        # Calculate reward
        reward = self.incentive_mechanism.calculate_reward(cid, score, contrib)
        client_rewards[cid] = reward
        
        # ADD: Distribute stake as reward (optional)
        if self.staking_system is not None:
            stake_reward = reward * 0.1  # 10% of reward as stake
            self.staking_system.add_stake(cid, stake_reward)
    
    self.incentive_mechanism.distribute_rewards(client_rewards)


# ============================================================================
# Step 8: Print Security Summary After Training
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
            print(f"  Client {event['client_id']}: Slashed {event['amount']:.2f} "
                  f"(Reason: {event['reason']}, Remaining: {event['remaining_balance']:.2f})")

