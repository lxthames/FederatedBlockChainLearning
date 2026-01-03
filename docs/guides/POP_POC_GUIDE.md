# Proof-of-Personhood (PoP) and Proof-of-Contribution (PoC) Guide

Complete guide to understanding and using PoP and PoC mechanisms in the federated learning framework.

## Table of Contents

- [Overview](#overview)
- [Proof-of-Personhood (PoP)](#proof-of-personhood-pop)
- [Proof-of-Contribution (PoC)](#proof-of-contribution-poc)
- [Integration](#integration)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## Overview

PoP and PoC are security and quality assurance mechanisms that ensure:
- **PoP**: Only real, verified participants can contribute
- **PoC**: Only useful, high-quality updates are accepted

Together, they create a secure, fair, and efficient federated learning system.

## Proof-of-Personhood (PoP)

### What is PoP?

Proof-of-Personhood prevents Sybil attacks by ensuring each participant is a real entity, not a fake identity created by an attacker.

### How It Works

#### Registration Process

1. **New Participant Arrives**
   ```
   Participant → Request Registration
   ```

2. **Choose Verification Method**
   - **Challenge-based**: Solve a puzzle (like CAPTCHA)
   - **Web-of-trust**: Get endorsements from existing members

3. **Complete Verification**
   - Challenge: Prove you're human/real
   - Web-of-trust: Get at least 2 endorsements

4. **Receive Credentials**
   - Unique verification token
   - Initial trust score

5. **Can Participate**
   - If trust score ≥ threshold (default: 0.5)
   - Can propose updates and vote

#### Trust Scoring

- **Initial Score**:
  - Challenge-based: 0.5
  - Web-of-trust: 1.0

- **Score Increases**:
  - With endorsements: `min(1.0, 0.5 + 0.1 * endorsements)`
  - With good behavior: (future enhancement)

- **Score Decreases**:
  - With bad behavior: (future enhancement)

#### Usage Example

```python
# Initialize PoP
pop = ProofOfPersonhood(min_trust_score=0.5)

# Register a node
success, token = pop.register_node(node_id=0, verification_method="challenge")
if success:
    print(f"Registered with token: {token}")

# Check if can participate
if pop.can_participate(node_id=0):
    print("Can participate in federated learning")
    
# Get trust score
score = pop.get_trust_score(node_id=0)
print(f"Trust score: {score}")
```

### Configuration

```python
# Minimum trust score to participate
min_trust_score = 0.5  # Range: 0.0 to 1.0

# Lower = more permissive (default: 0.5)
# Higher = more strict (e.g., 0.7)
```

## Proof-of-Contribution (PoC)

### What is PoC?

Proof-of-Contribution evaluates how useful each update is to the global model, ensuring only quality updates are accepted.

### How It Works

#### Evaluation Process

1. **Test Update on Validation Set**
   ```
   Local Model (with update) → Validation Set → Metrics
   Global Model → Validation Set → Baseline
   ```

2. **Calculate Metrics**
   - **Accuracy Improvement** (40% weight):
     - How much better is the updated model?
     - `max(0, global_loss - local_loss)`
   
   - **Loss Reduction** (40% weight):
     - Percentage improvement in loss
     - `(global_loss - local_loss) / (global_loss + ε)`
   
   - **Robustness Score** (20% weight):
     - Is the update stable?
     - `1.0 / (1.0 + update_norm / 100.0)`

3. **Compute Overall Score**
   ```
   overall_score = 0.4 × accuracy_improvement +
                   0.4 × loss_reduction +
                   0.2 × robustness_score
   ```

4. **Filter by Threshold**
   - If `overall_score >= threshold`: Accept
   - If `overall_score < threshold`: Reject

5. **Rank Updates** (optional)
   - Sort by score (highest first)
   - Select top-K if configured

#### Usage Example

```python
# Initialize PoC
poc = ProofOfContribution(validation_data=(X_val, Y_val))

# Evaluate an update
contribution = poc.evaluate_contribution(
    client_id=0,
    model_update=update_dict,
    local_model=local_model,
    global_model=global_model,
    validation_data=(X_val, Y_val),
    device=device
)

print(f"Accuracy improvement: {contribution['accuracy_improvement']}")
print(f"Loss reduction: {contribution['loss_reduction']}")
print(f"Robustness: {contribution['robustness_score']}")
print(f"Overall score: {contribution['overall_score']}")

# Check threshold
if poc.meets_threshold(contribution['overall_score'], threshold=0.1):
    print("Update accepted!")
else:
    print("Update rejected (below threshold)")

# Rank updates
ranked = poc.rank_updates(contributions, top_k=5)
print(f"Top 5 updates: {ranked}")
```

### Configuration

```python
# Minimum contribution score
poc_threshold = 0.1  # Range: 0.0 to 1.0

# Lower = more permissive (accepts more updates)
# Higher = more strict (only high-quality updates)

# Top-K filtering (optional)
top_k_updates = None  # None = all above threshold
top_k_updates = 5      # Only top 5 updates
```

## Integration

### Complete Workflow

```
1. Client Registration (PoP)
   ├─ Register with challenge or web-of-trust
   ├─ Receive trust score
   └─ Can participate if score ≥ threshold

2. Local Training
   ├─ Train model locally
   └─ Compute update

3. PoP Verification
   ├─ Check if client is verified
   └─ Reject if not verified

4. PoC Evaluation
   ├─ Evaluate update on validation set
   ├─ Calculate contribution score
   └─ Filter by threshold

5. Ranking (optional)
   ├─ Sort by score
   └─ Select top-K

6. Committee Validation
   ├─ Committee votes on PoC-accepted updates
   └─ Only PoP-verified members can vote

7. Blockchain Recording
   ├─ Record accepted updates
   └─ Include PoC metrics

8. Model Aggregation
   ├─ Aggregate validated updates
   └─ Update global model
```

### Integration Example

```python
# Initialize system with PoP/PoC
fl_system = EnhancedBlockchainFederatedLearning(
    num_clients=10,
    committee_size=3,
    input_size=128,
    num_outputs=5,
    device=device,
    validation_data=(X_val, Y_val),  # For PoC
    poc_threshold=0.1,               # PoC threshold
    top_k_updates=None                # All above threshold
)

# Initialize clients (PoP registration happens here)
fl_system.initialize_clients(client_data_list)

# Run federated round (PoP/PoC integrated)
stats = fl_system.federated_round_async(
    local_epochs=3,
    batch_size=128,
    lr=0.001,
    max_staleness=3
)
```

## Configuration

### Recommended Settings

#### For Development/Testing
```python
min_trust_score = 0.3      # More permissive
poc_threshold = 0.05       # Accept most updates
top_k_updates = None       # All above threshold
```

#### For Production
```python
min_trust_score = 0.5      # Standard
poc_threshold = 0.1        # Quality control
top_k_updates = None       # All above threshold
```

#### For High-Quality Only
```python
min_trust_score = 0.7      # Strict
poc_threshold = 0.3        # High quality only
top_k_updates = 5          # Top 5 updates
```

### Tuning Guidelines

1. **Start Permissive**: Low thresholds initially
2. **Monitor Results**: Check update quality
3. **Adjust Gradually**: Increase thresholds if needed
4. **Balance**: Too strict = few updates, too permissive = low quality

## Troubleshooting

### PoP Issues

#### Issue: Clients Not Registering

**Problem**: Registration fails

**Solutions**:
- Check verification method
- Ensure first 3 nodes use web-of-trust (auto-register)
- Verify challenge completion

#### Issue: Low Trust Scores

**Problem**: Trust scores below threshold

**Solutions**:
- Lower `min_trust_score`
- Get more endorsements (web-of-trust)
- Check registration status

### PoC Issues

#### Issue: All Updates Rejected

**Problem**: PoC threshold too high

**Solutions**:
- Lower `poc_threshold` (e.g., 0.05)
- Check validation data quality
- Verify model updates are computed correctly

#### Issue: Low Contribution Scores

**Problem**: Updates not improving model

**Solutions**:
- Check training quality
- Verify data quality
- Adjust learning rate
- Increase local epochs

#### Issue: Validation Data Missing

**Problem**: PoC cannot evaluate

**Solutions**:
- Provide validation data during initialization
- System falls back to simple metrics if missing
- Ensure validation data is appropriate

## Best Practices

### PoP Best Practices

1. **Trust Threshold**: Start with 0.5, adjust based on network
2. **Verification Method**: Use challenge for new networks, web-of-trust for established
3. **Monitor Registration**: Track registration success rate
4. **Trust Scores**: Monitor trust score distribution

### PoC Best Practices

1. **Threshold**: Start with 0.1, adjust based on update quality
2. **Validation Data**: Use representative validation set
3. **Top-K**: Use None initially, set top-K if too many updates
4. **Monitor Scores**: Track contribution score distribution
5. **Balance**: Balance between quality and quantity

## Related Documentation

- [Architecture Overview](../architecture/ARCHITECTURE.md)
- [Usage Guide](USAGE_GUIDE.md)
- [API Reference](../api/API_REFERENCE.md)
- [PoP/PoC Implementation Summary](../../PoP_PoC_Implementation_Summary.md)

