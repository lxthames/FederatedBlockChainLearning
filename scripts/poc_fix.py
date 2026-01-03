"""
Quick Fix for PoC Rejection Issue

Add this code to your notebook to fix the "0/X updates passed threshold" problem.
"""

# ============================================================================
# FIX 1: Add Diagnostic Output in federated_round_async()
# ============================================================================
# Add this RIGHT AFTER poc_contributions is calculated (around line 1280):

# DIAGNOSTIC: Print PoC scores to see why they're failing
print(f"\n[DEBUG] PoC Evaluation Results:")
for client_id, contrib in poc_contributions.items():
    print(f"  Client {client_id}:")
    print(f"    Accuracy improvement: {contrib['accuracy_improvement']:.6f}")
    print(f"    Loss reduction: {contrib['loss_reduction']:.6f}")
    print(f"    Robustness score: {contrib['robustness_score']:.6f}")
    print(f"    Overall score: {contrib['overall_score']:.6f}")
    print(f"    Threshold ({self.poc_threshold:.4f}): {'✓ PASS' if contrib['overall_score'] >= self.poc_threshold else '✗ FAIL'}")

# This will show you the actual scores


# ============================================================================
# FIX 2: Update Training Loop with Adaptive Threshold
# ============================================================================
# Replace your training loop with this:

for round_num in range(1, FEDERATED_ROUNDS + 1):
    print(f"\n--- Federated Round {round_num}/{FEDERATED_ROUNDS} ---")
    
    # Use adaptive threshold (starts low, increases over time)
    if round_num <= 3:
        adaptive_threshold = 0.01  # Very permissive early
    elif round_num <= 10:
        # Gradually increase
        progress = (round_num - 3) / 7
        adaptive_threshold = 0.01 + (0.1 - 0.01) * progress
    else:
        adaptive_threshold = 0.1  # Full threshold
    
    fl_system.poc_threshold = adaptive_threshold
    print(f"PoC threshold: {adaptive_threshold:.4f}")
    
    round_stats = fl_system.federated_round_async(
        local_epochs=LOCAL_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        max_staleness=3
    )
    print(f"Validated updates: {round_stats['validated_updates']}/{round_stats['total_updates']}")
    print(f"Block index: {round_stats['block_index']}")
    print(f"Consensus achieved: {round_stats['consensus_achieved']}")


# ============================================================================
# FIX 3: Alternative - Lower Base Threshold
# ============================================================================
# If adaptive threshold doesn't work, just lower the base threshold:

fl_system = EnhancedBlockchainFederatedLearning(
    ...
    poc_threshold=0.01,  # Lower from 0.1 to 0.01
    ...
)


# ============================================================================
# FIX 4: Check if Model is Learning
# ============================================================================
# Add this to verify the model is actually improving:

previous_val_loss = None
for round_num in range(1, FEDERATED_ROUNDS + 1):
    # ... training ...
    
    # After round, check improvement
    val_loss = evaluate_global_model(fl_system.global_model, X_val, Y_val, device)
    
    if previous_val_loss is not None:
        improvement = previous_val_loss - val_loss
        print(f"Validation loss improvement: {improvement:.6f}")
        if improvement < -0.001:  # Loss increased
            print("⚠️ WARNING: Model loss is increasing!")
            print("   Consider: increasing local epochs, adjusting learning rate, or checking data")
    
    previous_val_loss = val_loss

