"""
Integration Guide: How to integrate MetricsLogger into federated_round_async

This shows the exact hooks to add at each measurement point.
"""

# ============================================================================
# Step 1: Initialize MetricsLogger (at the start of main_blockchain_federated)
# ============================================================================

# In main_blockchain_federated(), after loading test data:
from scripts.metrics_logger import MetricsLogger

# Split test data (if not already split)
X_train, X_test, Y_train, Y_test = train_test_split(
    X_full, Y_full, test_size=0.1, random_state=42
)

# Initialize metrics logger
metrics_logger = MetricsLogger(
    num_clients=num_clients,
    test_data=(X_test, Y_test)  # Separate test set
)

# Set Sybil clients (if you have ground truth)
# Example: clients 2, 5, 8 are Sybil nodes
sybil_client_ids = [2, 5, 8]  # Replace with your actual Sybil IDs
metrics_logger.set_sybil_clients(sybil_client_ids)


# ============================================================================
# Step 2: Add Hook A - Utility Metrics (Post-Aggregation)
# Location: After global model update (Equation 12)
# ============================================================================

# In federated_round_async(), after aggregate_updates_async():
def federated_round_async_with_metrics(self, local_epochs: int = 1, batch_size: int = 128,
                                      lr: float = 0.001, max_staleness: int = 3,
                                      metrics_logger=None) -> Dict:
    # ... existing code for local training, PoP, PoC, etc. ...
    
    # After aggregation (Equation 12):
    if validated_updates:
        self.aggregate_updates_async(validated_updates)
        new_version = self.model_version.increment_version()
        self.model_version.store_version(new_version, self.global_model.state_dict())
        self.storage_optimizer.store_model(new_version, self.global_model.state_dict())
        self._update_contributions_and_rewards(validated_updates, validation_scores)
        
        # ====================================================================
        # HOOK A: Log Global Performance (Post-Aggregation)
        # ====================================================================
        if metrics_logger is not None:
            metrics_logger.log_global_performance(
                model=self.global_model,
                device=self.device,
                criterion=torch.nn.MSELoss(),
                batch_size=256
            )
        
        # ... rest of code ...


# ============================================================================
# Step 3: Add Hook B - Security Metrics (Post-Consensus)
# Location: Inside consensus block where P_accepted is finalized
# ========================================================================

# In federated_round_async(), after PoC filtering and before block mining:
def federated_round_async_with_metrics(self, ...):
    # ... existing code ...
    
    # After PoC filtering and committee validation:
    accepted_client_ids = [u[0]["client_id"] for u in validated_updates]
    
    # ====================================================================
    # HOOK B: Log Security Metrics (Post-Consensus)
    # ====================================================================
    if metrics_logger is not None:
        # poc_scores should be available from the PoC evaluation step
        metrics_logger.log_security_metrics(
            accepted_client_ids=accepted_client_ids,
            proof_of_personhood=self.proof_of_personhood,
            poc_scores=poc_scores  # Dictionary from PoC evaluation
        )
    
    # ... continue with block mining ...


# ============================================================================
# Step 4: Add Hook C - Consensus Efficiency (During Block Creation)
# Location: Inside block generation logic
# ========================================================================

# In federated_round_async(), during block creation:
def federated_round_async_with_metrics(self, ...):
    # ... existing code ...
    
    # When creating the block:
    committee_members = self.committee.committee_members
    
    # Extract staleness values from validated updates
    staleness_values = [u[0].get("staleness", 0) for u in validated_updates]
    
    # ====================================================================
    # HOOK C: Log Consensus Metrics (During Block Creation)
    # ====================================================================
    if metrics_logger is not None:
        metrics_logger.log_consensus_metrics(
            committee_members=committee_members,
            accepted_updates=[u[0] for u in validated_updates],  # List of update dicts
            staleness_values=staleness_values
        )
    
    # ... continue with block mining ...


# ============================================================================
# Step 5: Update Training Loop to Use MetricsLogger
# ========================================================================

# In main_blockchain_federated(), update the training loop:
def main_blockchain_federated():
    # ... existing setup code ...
    
    # Initialize metrics logger
    metrics_logger = MetricsLogger(num_clients=num_clients, test_data=(X_test, Y_test))
    metrics_logger.set_sybil_clients([2, 5, 8])  # Your Sybil IDs
    
    # Training loop
    for round_num in range(1, FEDERATED_ROUNDS + 1):
        print(f"\n--- Federated Round {round_num}/{FEDERATED_ROUNDS} ---")
        
        # Start new round in logger
        metrics_logger.start_new_round()
        
        # Run federated round with metrics
        round_stats = fl_system.federated_round_async(
            local_epochs=LOCAL_EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            max_staleness=3,
            metrics_logger=metrics_logger  # Pass logger
        )
        
        # ... existing code ...
    
    # After training: Generate plots and save metrics
    print("\n" + "="*70)
    print("GENERATING METRICS AND PLOTS")
    print("="*70)
    
    # Save metrics to JSON
    metrics_logger.save_metrics(f"{BASE_DIR}/metrics_round_{num_clients}_{dist_type}.json")
    
    # Generate all plots
    metrics_logger.plot_all(
        output_dir=f"{BASE_DIR}/metrics_plots_{num_clients}_{dist_type}",
        show=False  # Set to True to display plots
    )
    
    # Print summary
    summary = metrics_logger.get_summary()
    print("\n" + "="*70)
    print("METRICS SUMMARY")
    print("="*70)
    print(f"Final Test Recall: {summary['final_test_recall']*100:.2f}%")
    print(f"Final Test Loss: {summary['final_test_loss']:.6f}")
    print(f"Final Honest Trust: {summary['final_honest_trust']:.4f}")
    print(f"Final Sybil Trust: {summary['final_sybil_trust']:.4f}")
    print(f"Average ASR: {summary['average_asr']*100:.2f}%")
    print(f"Total Sybil Accepted: {summary['total_sybil_accepted']}")
    print(f"Total Sybil Proposed: {summary['total_sybil_proposed']}")
    print(f"Average Staleness: {summary['average_staleness']:.2f}")


# ============================================================================
# Step 6: Update EnhancedBlockchainFederatedLearning.federated_round_async
# ========================================================================

# Add metrics_logger parameter to the method signature:
class EnhancedBlockchainFederatedLearning:
    def federated_round_async(self, local_epochs: int = 1, batch_size: int = 128,
                              lr: float = 0.001, max_staleness: int = 3,
                              metrics_logger=None) -> Dict:  # Add this parameter
        # ... existing code ...
        
        # After aggregation (Hook A):
        if validated_updates:
            self.aggregate_updates_async(validated_updates)
            # ... existing code ...
            
            if metrics_logger is not None:
                metrics_logger.log_global_performance(
                    model=self.global_model,
                    device=self.device
                )
        
        # After PoC filtering (Hook B):
        accepted_client_ids = [u[0]["client_id"] for u in validated_updates]
        if metrics_logger is not None:
            metrics_logger.log_security_metrics(
                accepted_client_ids=accepted_client_ids,
                proof_of_personhood=self.proof_of_personhood,
                poc_scores=poc_scores  # Make sure poc_scores is available
            )
        
        # During block creation (Hook C):
        if self.blockchain.pending_transactions:
            committee_members = self.committee.committee_members
            staleness_values = [u[0].get("staleness", 0) for u in validated_updates]
            
            if metrics_logger is not None:
                metrics_logger.log_consensus_metrics(
                    committee_members=committee_members,
                    accepted_updates=[u[0] for u in validated_updates],
                    staleness_values=staleness_values
                )
            
            # ... continue with block mining ...

