# PoC Troubleshooting Guide

## Problem: All Updates Rejected (0/X updates passed threshold)

### Symptoms
- `PoC: 0/4 updates passed threshold`
- No blocks being created
- Model not learning (loss not improving)
- Stuck at genesis block

### Root Causes

1. **Threshold Too High**: Default threshold (0.1) might be too strict for early training
2. **Model Not Learning**: Updates aren't improving the model enough
3. **PoC Calculation Issue**: Scores might be calculated incorrectly
4. **Data Issues**: Model might not be able to learn from the data

## Solutions

### Solution 1: Lower Threshold (Quick Fix)

**In `main_blockchain_federated()`:**

```python
fl_system = EnhancedBlockchainFederatedLearning(
    ...
    poc_threshold=0.01,  # Lower from 0.1 to 0.01
    ...
)
```

**Or use adaptive threshold:**

```python
def get_adaptive_threshold(round_num, base_threshold=0.1, min_threshold=0.01):
    """Threshold starts low and increases over time"""
    if round_num <= 5:
        return min_threshold  # Very permissive early on
    elif round_num <= 10:
        return min_threshold + (base_threshold - min_threshold) * 0.5
    else:
        return base_threshold  # Full threshold later

# In training loop:
for round_num in range(1, FEDERATED_ROUNDS + 1):
    fl_system.poc_threshold = get_adaptive_threshold(round_num)
    round_stats = fl_system.federated_round_async(...)
```

### Solution 2: Add Diagnostic Output

**Add to `federated_round_async()` after PoC evaluation:**

```python
# After poc_contributions is calculated, add:
print(f"\n[DEBUG] PoC Scores:")
for client_id, contrib in poc_contributions.items():
    print(f"  Client {client_id}:")
    print(f"    Accuracy improvement: {contrib['accuracy_improvement']:.6f}")
    print(f"    Loss reduction: {contrib['loss_reduction']:.6f}")
    print(f"    Robustness: {contrib['robustness_score']:.6f}")
    print(f"    Overall: {contrib['overall_score']:.6f} (threshold: {self.poc_threshold})")
```

This will show you the actual scores and why they're failing.

### Solution 3: Check Model Learning

**Add validation check:**

```python
# After each round, check if model is learning:
previous_loss = None
for round_num in range(1, FEDERATED_ROUNDS + 1):
    # ... training ...
    
    # Check model improvement
    val_loss = evaluate_global_model(fl_system.global_model, X_val, Y_val, device)
    if previous_loss is not None:
        improvement = previous_loss - val_loss
        print(f"Loss improvement: {improvement:.6f}")
        if improvement < 0:
            print("⚠️ Model loss increasing - check learning rate or data!")
    previous_loss = val_loss
```

### Solution 4: Increase Local Epochs

**If model isn't learning enough per round:**

```python
LOCAL_EPOCHS = 5  # Increase from 3 to 5 or more
```

More local training = better updates = higher PoC scores.

### Solution 5: Adjust Learning Rate

**If model is learning too slowly:**

```python
LEARNING_RATE = 0.01  # Increase from 0.001
```

Or use learning rate scheduling.

### Solution 6: Check Data Quality

**Verify data is properly normalized and formatted:**

```python
# Check data statistics
print(f"X_train stats: min={X_train.min():.4f}, max={X_train.max():.4f}, mean={X_train.mean():.4f}")
print(f"Y_train stats: min={Y_train.min():.4f}, max={Y_train.max():.4f}, mean={Y_train.mean():.4f}")

# Check if Y is properly normalized (should sum to ~1.0 per row)
row_sums = Y_train.sum(axis=1)
print(f"Y row sums: min={row_sums.min():.4f}, max={row_sums.max():.4f}, mean={row_sums.mean():.4f}")
```

## Recommended Quick Fix

For immediate resolution, use **Solution 1** with adaptive threshold:

```python
# In main_blockchain_federated(), before training loop:
def get_adaptive_threshold(round_num):
    if round_num <= 5:
        return 0.01  # Very permissive early
    elif round_num <= 10:
        return 0.05  # Medium
    else:
        return 0.1   # Full threshold

# In training loop:
for round_num in range(1, FEDERATED_ROUNDS + 1):
    fl_system.poc_threshold = get_adaptive_threshold(round_num)
    print(f"Using PoC threshold: {fl_system.poc_threshold}")
    
    round_stats = fl_system.federated_round_async(
        local_epochs=LOCAL_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        max_staleness=3
    )
```

This allows updates early in training when the model is still learning, then becomes stricter as training progresses.

## Diagnostic Checklist

- [ ] Check PoC scores with debug output
- [ ] Verify threshold value (might be too high)
- [ ] Check if model loss is decreasing
- [ ] Verify data is properly formatted
- [ ] Check learning rate (might be too low)
- [ ] Increase local epochs if needed
- [ ] Verify PoC evaluation is working correctly

## Expected PoC Scores

Early in training, PoC scores are typically:
- **Round 1-3**: 0.01 - 0.05 (very low, model just starting)
- **Round 4-10**: 0.05 - 0.15 (improving)
- **Round 10+**: 0.10 - 0.30+ (good updates)

If scores are consistently < 0.01, the model might not be learning at all.

