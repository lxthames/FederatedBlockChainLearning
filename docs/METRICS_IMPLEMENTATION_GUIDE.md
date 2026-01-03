# Metrics & Reporting Implementation Guide

## Overview

This guide explains how to implement a centralized logging system to track:
1. **Model Utility**: Test accuracy and loss over T rounds
2. **Security**: Trust scores and attack success rate (Sybil resistance)
3. **Consensus Efficiency**: Committee participation and staleness

## Files Created

1. **`scripts/metrics_logger.py`**: Complete MetricsLogger class
2. **`scripts/metrics_integration.py`**: Integration examples
3. **This guide**: Step-by-step instructions

## Quick Start

### Step 1: Import MetricsLogger

Add to your imports cell:

```python
from scripts.metrics_logger import MetricsLogger
import matplotlib.pyplot as plt
```

### Step 2: Initialize in main_blockchain_federated()

```python
def main_blockchain_federated():
    # ... existing setup code ...
    
    # Split data: train, val, test
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X_full, Y_full, test_size=0.2, random_state=42
    )
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp, Y_temp, test_size=0.5, random_state=42
    )
    
    # Initialize metrics logger
    metrics_logger = MetricsLogger(
        num_clients=num_clients,
        test_data=(X_test, Y_test)  # Separate test set (not used by clients)
    )
    
    # Set Sybil clients (ground truth - replace with your actual Sybil IDs)
    sybil_client_ids = [2, 5, 8]  # Example: clients 2, 5, 8 are Sybil
    metrics_logger.set_sybil_clients(sybil_client_ids)
    
    # ... rest of setup ...
```

### Step 3: Update federated_round_async Method Signature

Add `metrics_logger` parameter:

```python
def federated_round_async(self, local_epochs: int = 1, batch_size: int = 128,
                          lr: float = 0.001, max_staleness: int = 3,
                          metrics_logger=None) -> Dict:  # Add this
```

### Step 4: Add Hook A - Global Performance (Post-Aggregation)

**Location**: After `aggregate_updates_async()` call

```python
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
```

### Step 5: Add Hook B - Security Metrics (Post-Consensus)

**Location**: After PoC filtering, before block mining

```python
# After PoC evaluation and filtering:
accepted_client_ids = [u[0]["client_id"] for u in validated_updates]

# ====================================================================
# HOOK B: Log Security Metrics (Post-Consensus)
# ====================================================================
if metrics_logger is not None:
    # Make sure poc_scores is available (from PoC evaluation step)
    metrics_logger.log_security_metrics(
        accepted_client_ids=accepted_client_ids,
        proof_of_personhood=self.proof_of_personhood,
        poc_scores=poc_scores  # Dictionary from PoC evaluation
    )
```

**Note**: You need to make `poc_scores` available. In the refactored version, this should be the dictionary created during committee voting.

### Step 6: Add Hook C - Consensus Efficiency (During Block Creation)

**Location**: When creating the block, before `mine_block()`

```python
# When creating the block:
if self.blockchain.pending_transactions:
    committee_members = self.committee.committee_members
    
    # Extract staleness values
    staleness_values = [u[0].get("staleness", 0) for u in validated_updates]
    
    # ====================================================================
    # HOOK C: Log Consensus Metrics (During Block Creation)
    # ====================================================================
    if metrics_logger is not None:
        metrics_logger.log_consensus_metrics(
            committee_members=committee_members,
            accepted_updates=[u[0] for u in validated_updates],
            staleness_values=staleness_values
        )
    
    # Continue with block mining
    block_votes = {m: len(validated_updates) > 0 for m in committee_members}
    self.blockchain.mine_block(block_votes)
```

### Step 7: Update Training Loop

```python
# In main_blockchain_federated(), training loop:
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
```

### Step 8: Generate Plots and Save Metrics (After Training)

```python
# After training loop completes:
print("\n" + "="*70)
print("GENERATING METRICS AND PLOTS")
print("="*70)

# Save metrics to JSON
metrics_logger.save_metrics(f"{BASE_DIR}/metrics_round_{num_clients}_{dist_type}.json")

# Generate all plots
metrics_logger.plot_all(
    output_dir=f"{BASE_DIR}/metrics_plots_{num_clients}_{dist_type}",
    show=False  # Set to True to display plots interactively
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
print(f"Total Sybil Accepted: {summary['total_sybil_accepted']}")
print(f"Total Sybil Proposed: {summary['total_sybil_proposed']}")
print(f"Average Staleness: {summary['average_staleness']:.2f}")
```

## Metrics Tracked

### 1. Global Performance
- **test_accuracy_history**: Test accuracy at each round
- **test_loss_history**: Test loss F(w) at each round

### 2. Security Metrics
- **trust_scores_honest**: Average τ for honest nodes per round
- **trust_scores_sybil**: Average τ for Sybil nodes per round
- **attack_success_rate**: ASR = Sybil Accepted / Sybil Proposed per round

### 3. Consensus Efficiency
- **committee_participation**: Dictionary of client_id → selection count
- **average_staleness**: Average Δt_k for accepted updates per round

## Generated Plots

### Plot 1: Global Convergence
- **X-Axis**: Communication Round (t)
- **Y-Axis**: Test Accuracy (%)
- **Purpose**: Prove system learns effectively

### Plot 2: Trust Dynamics (The "Money Plot")
- **X-Axis**: Communication Round (t)
- **Y-Axis**: Average Trust Score (τ)
- **Lines**: 
  - Honest Nodes (should converge to 1.0)
  - Sybil Nodes (should converge to 0 or τ_min)
- **Purpose**: Demonstrate Sybil resistance

### Plot 3: Committee Fairness Histogram
- **X-Axis**: Client IDs
- **Y-Axis**: Number of Selections
- **Colors**: Green (honest), Red (Sybil)
- **Purpose**: Demonstrate election probability prevents centralization

## Example Output

After running, you'll get:

1. **JSON file**: `metrics_round_10_skewed.json` with all raw data
2. **Plot files**:
   - `global_convergence.png`
   - `trust_dynamics.png`
   - `committee_fairness.png`
3. **Console summary** with key metrics

## Troubleshooting

### "poc_scores not defined"
- Make sure you're storing `poc_scores` dictionary during PoC evaluation
- Pass it to `log_security_metrics()`

### "No test data available"
- Make sure you split data into train/val/test
- Pass test data to `MetricsLogger(test_data=(X_test, Y_test))`

### "Sybil clients not set"
- Call `metrics_logger.set_sybil_clients([...])` after initialization
- Replace with your actual Sybil client IDs

### Plots not generating
- Check that you have data in the history lists
- Verify matplotlib is installed: `pip install matplotlib`

## Advanced Usage

### Custom Sybil Detection

If you don't have ground truth, you can identify Sybils based on behavior:

```python
# Example: Identify Sybils based on low trust scores
low_trust_clients = [
    cid for cid in range(num_clients)
    if proof_of_personhood.get_trust_score(cid) < 0.3
]
metrics_logger.set_sybil_clients(low_trust_clients)
```

### Export Metrics for Analysis

```python
# Save to CSV for external analysis
import pandas as pd

df = pd.DataFrame({
    'round': range(1, len(metrics_logger.test_accuracy_history) + 1),
    'test_accuracy': metrics_logger.test_accuracy_history,
    'test_loss': metrics_logger.test_loss_history,
    'honest_trust': metrics_logger.trust_scores_honest,
    'sybil_trust': metrics_logger.trust_scores_sybil,
    'asr': metrics_logger.attack_success_rate,
    'staleness': metrics_logger.average_staleness
})
df.to_csv('metrics.csv', index=False)
```

## Mathematical Formulation

### Attack Success Rate (ASR)

$$ASR^{(t)} = \frac{\text{Sybil Updates Accepted in Block } B_t}{\text{Sybil Updates Proposed (passed } \delta_{th})}$$

### Average Staleness

$$\bar{\Delta t}^{(t)} = \frac{1}{|P_{accepted}|} \sum_{k \in P_{accepted}} \Delta t_k$$

Where $\Delta t_k = t_{current} - t_{trained}$.

### Committee Participation

For each client $i$:
$$N_{committee}(i) = \sum_{t=1}^{T} \mathbb{1}[i \in C^{(t)}]$$

Where $C^{(t)}$ is the committee at round $t$.

## Verification Checklist

- [ ] MetricsLogger initialized with test data
- [ ] Sybil clients set correctly
- [ ] Hook A added after aggregation
- [ ] Hook B added after PoC filtering
- [ ] Hook C added during block creation
- [ ] Training loop passes metrics_logger
- [ ] Plots generated successfully
- [ ] Metrics saved to JSON
- [ ] Summary printed correctly

## Next Steps

1. Run training with metrics enabled
2. Review generated plots
3. Analyze JSON metrics file
4. Compare metrics across different configurations
5. Use metrics for paper/report generation

