"""
Quick Reference: Metrics Integration Code Snippets

Copy and paste these snippets into your notebook at the specified locations.
"""

# ============================================================================
# SNIPPET 1: Initialize MetricsLogger (in main_blockchain_federated)
# ============================================================================
# Add after data loading and splitting:

from scripts.metrics_logger import MetricsLogger

# Split data into train, val, test
X_train, X_temp, Y_train, Y_temp = train_test_split(
    X_full, Y_full, test_size=0.2, random_state=42
)
X_val, X_test, Y_val, Y_test = train_test_split(
    X_temp, Y_temp, test_size=0.5, random_state=42
)

# Initialize metrics logger
metrics_logger = MetricsLogger(
    num_clients=num_clients,
    test_data=(X_test, Y_test)
)

# Set Sybil clients (replace with your actual Sybil IDs)
sybil_client_ids = [2, 5, 8]  # Example
metrics_logger.set_sybil_clients(sybil_client_ids)


# ============================================================================
# SNIPPET 2: Update federated_round_async Signature
# ============================================================================
# Change method signature to:

def federated_round_async(self, local_epochs: int = 1, batch_size: int = 128,
                          lr: float = 0.001, max_staleness: int = 3,
                          metrics_logger=None) -> Dict:  # ADD THIS PARAMETER


# ============================================================================
# SNIPPET 3: Hook A - Global Performance (After Aggregation)
# ============================================================================
# Add immediately after aggregate_updates_async():

if validated_updates:
    self.aggregate_updates_async(validated_updates)
    new_version = self.model_version.increment_version()
    self.model_version.store_version(new_version, self.global_model.state_dict())
    self.storage_optimizer.store_model(new_version, self.global_model.state_dict())
    self._update_contributions_and_rewards(validated_updates, validation_scores)
    
    # HOOK A: Log Global Performance
    if metrics_logger is not None:
        metrics_logger.log_global_performance(
            model=self.global_model,
            device=self.device,
            criterion=torch.nn.MSELoss(),
            batch_size=256
        )


# ============================================================================
# SNIPPET 4: Hook B - Security Metrics (After PoC Filtering)
# ============================================================================
# Add after PoC evaluation, before block mining:

# After PoC filtering:
accepted_client_ids = [u[0]["client_id"] for u in validated_updates]

# HOOK B: Log Security Metrics
if metrics_logger is not None:
    metrics_logger.log_security_metrics(
        accepted_client_ids=accepted_client_ids,
        proof_of_personhood=self.proof_of_personhood,
        poc_scores=poc_scores  # Make sure this is available from PoC step
    )


# ============================================================================
# SNIPPET 5: Hook C - Consensus Efficiency (During Block Creation)
# ============================================================================
# Add when creating the block, before mine_block():

if self.blockchain.pending_transactions:
    committee_members = self.committee.committee_members
    staleness_values = [u[0].get("staleness", 0) for u in validated_updates]
    
    # HOOK C: Log Consensus Metrics
    if metrics_logger is not None:
        metrics_logger.log_consensus_metrics(
            committee_members=committee_members,
            accepted_updates=[u[0] for u in validated_updates],
            staleness_values=staleness_values
        )
    
    # Continue with block mining
    block_votes = {m: len(validated_updates) > 0 for m in committee_members}
    self.blockchain.mine_block(block_votes)


# ============================================================================
# SNIPPET 6: Update Training Loop
# ============================================================================
# In main_blockchain_federated(), update the loop:

for round_num in range(1, FEDERATED_ROUNDS + 1):
    print(f"\n--- Federated Round {round_num}/{FEDERATED_ROUNDS} ---")
    
    # Start new round
    metrics_logger.start_new_round()
    
    # Run federated round
    round_stats = fl_system.federated_round_async(
        local_epochs=LOCAL_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        max_staleness=3,
        metrics_logger=metrics_logger  # ADD THIS
    )
    
    # ... existing code ...


# ============================================================================
# SNIPPET 7: Generate Plots and Save (After Training)
# ============================================================================
# Add after training loop completes:

print("\n" + "="*70)
print("GENERATING METRICS AND PLOTS")
print("="*70)

# Save metrics
metrics_logger.save_metrics(f"{BASE_DIR}/metrics_{num_clients}_{dist_type}.json")

# Generate plots
metrics_logger.plot_all(
    output_dir=f"{BASE_DIR}/metrics_plots_{num_clients}_{dist_type}",
    show=False
)

# Print summary
summary = metrics_logger.get_summary()
print("\n" + "="*70)
print("METRICS SUMMARY")
print("="*70)
print(f"Final Test Accuracy: {summary['final_test_accuracy']*100:.2f}%")
print(f"Final Test Loss: {summary['final_test_loss']:.6f}")
print(f"Final Honest Trust: {summary['final_honest_trust']:.4f}")
print(f"Final Sybil Trust: {summary['final_sybil_trust']:.4f}")
print(f"Average ASR: {summary['average_asr']*100:.2f}%")
print(f"Average Staleness: {summary['average_staleness']:.2f}")

