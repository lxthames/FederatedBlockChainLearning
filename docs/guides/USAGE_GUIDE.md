# Usage Guide

Complete guide for using the Blockchain-Based Decentralized Federated Learning Framework.

## Table of Contents

- [Quick Start](#quick-start)
- [Basic Usage](#basic-usage)
- [Advanced Usage](#advanced-usage)
- [Configuration Options](#configuration-options)
- [Examples](#examples)
- [Best Practices](#best-practices)

## Quick Start

### 1. Open the Notebook

```bash
jupyter notebook Reduced_Layers_MUltihead.ipynb
```

### 2. Run Basic Training

```python
# Execute the main function
main_blockchain_federated()
```

### 3. View Results

- Check console output for training progress
- Inspect `blockchain_state_*.json` files
- Review saved model checkpoints

## Basic Usage

### Running Federated Learning

```python
from Reduced_Layers_MUltihead import main_blockchain_federated

# Run with default settings
main_blockchain_federated()
```

### Custom Configuration

```python
# Modify parameters in the notebook before running
# Or create custom configuration

config = {
    "num_clients": 10,
    "committee_size": 3,
    "local_epochs": 3,
    "batch_size": 128,
    "learning_rate": 0.001,
    "federated_rounds": 10,
    "poc_threshold": 0.1
}
```

## Advanced Usage

### Custom Model

```python
from Reduced_Layers_MUltihead import EnhancedBlockchainFederatedLearning
import torch

# Create custom model
class CustomModel(nn.Module):
    def __init__(self, input_size, num_outputs):
        super().__init__()
        # Your model architecture
        pass
    
    def forward(self, x):
        # Your forward pass
        pass

# Use custom model
fl_system = EnhancedBlockchainFederatedLearning(
    num_clients=10,
    committee_size=3,
    input_size=128,
    num_outputs=5,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)
```

### Custom PoP/PoC Settings

```python
# Initialize with custom PoP/PoC
fl_system = EnhancedBlockchainFederatedLearning(
    num_clients=10,
    committee_size=3,
    input_size=128,
    num_outputs=5,
    device=device,
    validation_data=(X_val, Y_val),
    poc_threshold=0.15,  # Higher threshold
    top_k_updates=5      # Only top 5 updates
)

# Custom PoP trust score
fl_system.proof_of_personhood.min_trust_score = 0.7
```

### Manual Round Execution

```python
# Initialize system
fl_system = EnhancedBlockchainFederatedLearning(...)
fl_system.initialize_clients(client_data_list)

# Run rounds manually
for round_num in range(10):
    print(f"Round {round_num + 1}")
    
    # Execute round
    stats = fl_system.federated_round_async(
        local_epochs=3,
        batch_size=128,
        lr=0.001,
        max_staleness=3
    )
    
    # Print statistics
    print(f"Validated: {stats['validated_updates']}/{stats['total_updates']}")
    
    # Start new round
    fl_system.start_new_round()
```

## Configuration Options

### Federated Learning Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `num_clients` | Number of clients | 10 | 2-100+ |
| `committee_size` | Committee size | 25% of clients | 1 to num_clients |
| `local_epochs` | Local training epochs | 3 | 1-10 |
| `batch_size` | Batch size | 128 | 32-512 |
| `learning_rate` | Learning rate | 0.001 | 0.0001-0.01 |
| `federated_rounds` | Number of rounds | 10 | 1-100+ |
| `max_staleness` | Max allowed staleness | 3 | 1-10 |

### PoP Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `min_trust_score` | Minimum trust to participate | 0.5 | 0.0-1.0 |
| `verification_method` | "challenge" or "web_of_trust" | "challenge" | - |

### PoC Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `poc_threshold` | Minimum contribution score | 0.1 | 0.0-1.0 |
| `top_k_updates` | Top-K updates to accept | None | 1 to num_clients |

### Committee Parameters

| Parameter | Description | Options |
|-----------|-------------|---------|
| `election_method` | Committee election method | "score", "random", "multi_factor" |

## Examples

### Example 1: Basic Training

```python
# Simple training run
main_blockchain_federated()
```

### Example 2: Custom Client Count

```python
# Train with 20 clients
# Modify in notebook:
CLIENT_SCALES = [20]  # Instead of [4, 8, 12, 16]
main_blockchain_federated()
```

### Example 3: High-Quality Updates Only

```python
# Only accept high-quality updates
fl_system = EnhancedBlockchainFederatedLearning(
    num_clients=10,
    committee_size=3,
    input_size=128,
    num_outputs=5,
    device=device,
    poc_threshold=0.3,  # Higher threshold
    top_k_updates=3     # Only top 3
)
```

### Example 4: Monitoring Training

```python
# Track training progress
fl_system = EnhancedBlockchainFederatedLearning(...)
fl_system.initialize_clients(client_data_list)

for round_num in range(10):
    stats = fl_system.federated_round_async(...)
    
    # Get blockchain info
    bc_info = fl_system.get_blockchain_info()
    print(f"Blocks: {bc_info['chain_length']}")
    print(f"Rewards: {bc_info['total_rewards']:.4f}")
    
    # Get client statistics
    client_stats = fl_system.get_client_statistics()
    top_clients = sorted(
        client_stats.items(), 
        key=lambda x: x[1]['total_contribution'], 
        reverse=True
    )[:3]
    print(f"Top contributors: {top_clients}")
```

### Example 5: Custom Data Distribution

```python
# Use custom data distribution
def custom_distribute_data(X, Y, num_clients):
    # Your custom distribution logic
    pass

# Use in notebook
client_data_list = custom_distribute_data(X_train, Y_train, num_clients)
```

## Best Practices

### 1. Data Preparation

- **Normalize data**: Ensure data is properly normalized
- **Split validation**: Always use validation set for PoC
- **Balance distribution**: Consider data distribution across clients

### 2. Model Configuration

- **Appropriate size**: Match model size to data size
- **Regularization**: Use dropout to prevent overfitting
- **Initialization**: Use proper weight initialization

### 3. Training Configuration

- **Learning rate**: Start with 0.001, adjust based on convergence
- **Batch size**: Balance memory and training speed
- **Local epochs**: 3-5 epochs typically sufficient
- **Rounds**: 10-20 rounds for good convergence

### 4. PoP/PoC Configuration

- **Trust threshold**: 0.5 is reasonable starting point
- **PoC threshold**: 0.1 allows most updates, 0.3 is stricter
- **Top-K**: Use None for all above threshold, or set to N for top-N

### 5. Committee Configuration

- **Size**: 20-30% of clients is typical
- **Election**: "score" method rewards good contributors
- **Rotation**: Automatic each round ensures fairness

### 6. Monitoring

- **Track metrics**: Monitor loss, accuracy, blockchain stats
- **Check logs**: Review console output regularly
- **Inspect blockchain**: Verify transactions are recorded
- **Monitor rewards**: Ensure fair distribution

### 7. Error Handling

- **Handle failures**: System handles client failures gracefully
- **Check staleness**: Monitor model staleness
- **Validate chain**: Check blockchain integrity
- **Backup models**: Save model checkpoints regularly

## Common Patterns

### Pattern 1: Early Stopping

```python
best_val_loss = float("inf")
patience = 5
patience_counter = 0

for round_num in range(100):
    stats = fl_system.federated_round_async(...)
    val_loss = evaluate_global_model(...)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(fl_system.global_model.state_dict(), "best_model.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping")
            break
```

### Pattern 2: Learning Rate Scheduling

```python
initial_lr = 0.001
for round_num in range(10):
    # Decay learning rate
    lr = initial_lr * (0.9 ** round_num)
    
    stats = fl_system.federated_round_async(
        local_epochs=3,
        batch_size=128,
        lr=lr
    )
```

### Pattern 3: Adaptive Threshold

```python
# Start with low threshold, increase over time
initial_threshold = 0.05
for round_num in range(10):
    threshold = initial_threshold + (round_num * 0.01)
    # Update PoC threshold (requires system modification)
```

## Related Documentation

- [Setup Guide](SETUP_GUIDE.md)
- [Architecture Overview](../architecture/ARCHITECTURE.md)
- [API Reference](../api/API_REFERENCE.md)
- [PoP/PoC Guide](POP_POC_GUIDE.md)

