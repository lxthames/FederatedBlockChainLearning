"""
PoC Debugging: Diagnose why updates are being rejected

Add this diagnostic code to understand PoC scores.
"""

# ============================================================================
# Diagnostic Code to Add in federated_round_async()
# ============================================================================

# After PoC evaluation, before filtering, add this:

# DIAGNOSTIC: Print PoC scores for debugging
print(f"\n[DEBUG] PoC Scores for Round {round_num}:")
for client_id, contrib in poc_contributions.items():
    print(f"  Client {client_id}:")
    print(f"    Accuracy improvement: {contrib['accuracy_improvement']:.6f}")
    print(f"    Loss reduction: {contrib['loss_reduction']:.6f}")
    print(f"    Robustness score: {contrib['robustness_score']:.6f}")
    print(f"    Overall score: {contrib['overall_score']:.6f}")
    print(f"    Threshold ({poc_threshold}): {'✓ PASS' if contrib['overall_score'] >= poc_threshold else '✗ FAIL'}")

# This will help you see why updates are being rejected


# ============================================================================
# Quick Fix: Lower Threshold Temporarily
# ============================================================================

# Option 1: Lower the threshold in main_blockchain_federated()
fl_system = EnhancedBlockchainFederatedLearning(
    ...
    poc_threshold=0.01,  # Lower from 0.1 to 0.01 for early training
    ...
)

# Option 2: Use adaptive threshold (starts low, increases over time)
def get_adaptive_threshold(round_num, base_threshold=0.1, min_threshold=0.01):
    """Threshold starts low and increases over time"""
    if round_num <= 5:
        return min_threshold
    elif round_num <= 10:
        return min_threshold + (base_threshold - min_threshold) * 0.5
    else:
        return base_threshold

# Use in training loop:
for round_num in range(1, FEDERATED_ROUNDS + 1):
    adaptive_threshold = get_adaptive_threshold(round_num)
    fl_system.poc_threshold = adaptive_threshold
    # ... rest of training ...


# ============================================================================
# Check Model Learning
# ============================================================================

# Add this to check if model is actually learning:
def check_model_learning(fl_system, X_val, Y_val, device):
    """Check if model is improving"""
    from scripts.metrics_logger import evaluate_global_model
    
    val_loss = evaluate_global_model(fl_system.global_model, X_val, Y_val, device)
    
    # Check if loss is decreasing
    if hasattr(fl_system, 'previous_val_loss'):
        improvement = fl_system.previous_val_loss - val_loss
        print(f"  Validation loss improvement: {improvement:.6f}")
        if improvement < 0:
            print("  ⚠️ Model loss is increasing - model may not be learning!")
    else:
        fl_system.previous_val_loss = val_loss
    
    return val_loss

