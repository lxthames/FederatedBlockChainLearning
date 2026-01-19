# Multihead Architecture in Federated Learning

## Overview

The **EnhancedMultiHeadDNN** (Multi-Head Deep Neural Network) is a specialized neural network architecture designed for the blockchain-based federated learning system. This document explains the multihead architecture, its design rationale, and how it integrates into the federated learning workflow.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Design Rationale](#design-rationale)
- [Architecture Components](#architecture-components)
- [Integration with Federated Learning](#integration-with-federated-learning)
- [Training Process](#training-process)
- [Inference and Predictions](#inference-and-predictions)
- [Benefits and Trade-offs](#benefits-and-trade-offs)
- [Code Examples](#code-examples)

## Architecture Overview

The `EnhancedMultiHeadDNN` employs a **parallel branch architecture** with **multi-head outputs** to enable efficient feature extraction and per-client predictions in a federated learning setting.

### Key Characteristics

- **Parallel Feature Extraction**: Two independent branches process input features simultaneously
- **Feature Merging**: Branch outputs are combined through weighted averaging
- **Multi-Head Output**: One output head per client, producing normalized probability distributions
- **Shallow Depth**: Limited to 3 layers to prevent overfitting and maintain efficiency
- **Regularization**: Batch normalization and dropout for stable training

## Design Rationale

### Why Parallel Branches?

**Decision**: Use two parallel branches instead of a single sequential network

**Rationale**:
1. **Diversity in Feature Extraction**: Each branch can learn different feature representations, capturing complementary information
2. **Stability**: Parallel processing reduces the risk of gradient vanishing/exploding in deeper networks
3. **Efficiency**: Parallel branches can be computed simultaneously, improving throughput
4. **Robustness**: Multiple feature extraction paths provide redundancy and improve generalization

### Why Multi-Head Output?

**Decision**: One output head per client instead of a single shared head

**Rationale**:
1. **Per-Client Predictions**: Each client can have different prediction requirements or data distributions
2. **Federated Learning Compatibility**: Aligns with the federated setting where each client contributes unique data
3. **Normalized Probabilities**: Softmax normalization ensures outputs are valid probability distributions
4. **Scalability**: Easy to add or remove clients by adjusting the number of heads

### Why Shallow Depth?

**Decision**: Limit network depth to ≤ 3 layers

**Rationale**:
1. **Prevent Overfitting**: Shallow networks are less prone to overfitting, especially with limited data per client
2. **Faster Convergence**: Fewer parameters mean faster training and aggregation
3. **Communication Efficiency**: Smaller models reduce communication overhead in federated learning
4. **Stability**: Shallow networks have more stable gradients during federated aggregation

## Architecture Components

### 1. Parallel Branches

The model contains two parallel branches that process the input independently:

```python
# Branch 1: Primary feature extraction
self.branch1 = nn.Sequential(
    nn.Linear(input_size, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(dropout_rate),
)

# Branch 2: Secondary feature extraction
self.branch2 = nn.Sequential(
    nn.Linear(input_size, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(dropout_rate * 0.5),  # Slightly less dropout
)
```

**Key Features**:
- Both branches map input to 128-dimensional feature space
- Batch normalization ensures stable training across different data distributions
- ReLU activation introduces non-linearity
- Dropout provides regularization (branch 2 has reduced dropout)

### 2. Feature Merging

The outputs from both branches are merged using weighted averaging:

```python
features = 0.5 * (f1 + f2)  # Simple average
```

**Why Averaging?**
- **Stability**: Averaging is more stable than concatenation (which would double the feature dimension)
- **Simplicity**: No additional parameters needed
- **Effectiveness**: Combines complementary features from both branches

### 3. Multi-Head Output Layer

Each client has its own dedicated output head:

```python
self.heads = nn.ModuleList([
    nn.Linear(128, 1)  # One head per client
    for _ in range(num_clients)
])
```

**Structure**:
- Each head is a simple linear layer mapping 128 features to 1 output
- The number of heads equals the number of clients in the federated system
- All heads share the same merged feature representation

### 4. Output Normalization

The final output is normalized using softmax:

```python
logits = torch.cat([head(features) for head in self.heads], dim=1)
return torch.softmax(logits, dim=1)
```

**Result**: 
- Output shape: `(batch_size, num_clients)`
- Each row sums to 1.0 (valid probability distribution)
- Each column represents the probability for a specific client

## Integration with Federated Learning

### 1. Model Initialization

The multihead model is initialized in the `EnhancedBlockchainFederatedLearning` class:

```python
# In federated_learning.py
self.global_model = EnhancedMultiHeadDNN(input_size, num_outputs).to(device)
```

**Parameters**:
- `input_size`: Dimension of input features (e.g., 128 for SIFT descriptors)
- `num_outputs`: Number of clients (determines number of heads)

### 2. Client Model Distribution

Each client receives a copy of the global model:

```python
# In initialize_clients()
client_model = copy.deepcopy(self.global_model)
client = FederatedClient(i, client_model, (X_train, Y_train), self.device)
```

**Important**: 
- All clients start with the same model architecture
- Each client trains on its local data
- Model updates are aggregated back to the global model

### 3. Local Training

During local training, each client:

1. **Receives Input**: Feature vectors (e.g., SIFT descriptors)
2. **Forward Pass**: 
   - Input → Branch 1 & Branch 2 (parallel)
   - Merge features: `0.5 * (branch1_out + branch2_out)`
   - Pass through client-specific head
   - Apply softmax normalization
3. **Computes Loss**: MSE loss between predictions and ground truth
4. **Backward Pass**: Updates all model parameters (branches + heads)

```python
# In FederatedClient.train_local()
preds = self.model(Xb)  # Shape: (batch_size, num_clients)
loss = criterion(preds, Yb)  # Yb shape: (batch_size, num_clients)
loss.backward()
```

### 4. Model Update Extraction

After local training, each client computes the difference between its local model and the global model:

```python
# In FederatedClient.get_model_update()
update = local_state[key] - global_model_state[key]
```

**Key Point**: The update includes changes to:
- Branch 1 parameters
- Branch 2 parameters  
- All client heads (including other clients' heads)

### 5. Aggregation

The global model aggregates updates from all validated clients:

```python
# In aggregate_updates_async()
for k in global_state:
    ws = torch.zeros_like(global_state[k])
    for update_info, _, _ in validated_updates:
        weight = fusion_weights.get(cid, 0.0)
        ws += weight * client_state[k]
    aggregated_state[k] = ws
```

**Aggregation Strategy**:
- Weighted averaging based on:
  - Data size contribution
  - Trust scores (probationary clients have reduced weight)
  - Staleness penalties (for async updates)
  - Contribution gaps

## Training Process

### Forward Pass Flow

```
Input (batch_size, input_size)
    ↓
    ├─→ Branch 1 → (batch_size, 128)
    │     ↓
    │   BatchNorm + ReLU + Dropout
    │
    └─→ Branch 2 → (batch_size, 128)
          ↓
        BatchNorm + ReLU + Dropout
          ↓
    Merge: 0.5 * (branch1 + branch2) → (batch_size, 128)
          ↓
    ┌─────────────────────────────────┐
    │  Head 0: Linear(128, 1)         │
    │  Head 1: Linear(128, 1)         │
    │  ...                             │
    │  Head N: Linear(128, 1)          │
    └─────────────────────────────────┘
          ↓
    Concatenate → (batch_size, num_clients)
          ↓
    Softmax → (batch_size, num_clients) [probabilities]
```

### Training Characteristics

1. **Shared Representation Learning**: 
   - Both branches learn shared feature representations
   - All heads share the same merged features
   - Enables knowledge transfer across clients

2. **Client-Specific Adaptation**:
   - Each head can specialize for its client's data distribution
   - Heads learn to extract client-relevant information from shared features

3. **Regularization**:
   - Batch normalization: Normalizes activations across batches
   - Dropout: Prevents overfitting by randomly zeroing activations
   - Weight initialization: Kaiming initialization for stable training

## Inference and Predictions

### Prediction Process

During inference, the model produces probability distributions over clients:

```python
# Example: Predicting which client a query belongs to
query = torch.tensor([[0.1, 0.2, ..., 0.9]])  # Shape: (1, input_size)
probs = model(query)  # Shape: (1, num_clients)
# Output: [[0.05, 0.15, 0.30, 0.20, 0.30]]  # Probabilities sum to 1.0
predicted_client = probs.argmax(dim=1)  # Client with highest probability
```

### Use Cases

1. **Client Identification**: Determine which client a query vector belongs to
2. **Similarity Matching**: Find clients with similar data distributions
3. **Anomaly Detection**: Low probabilities across all clients may indicate anomalies

## Benefits and Trade-offs

### Benefits

1. **Efficiency**: 
   - Parallel branches enable faster feature extraction
   - Shallow depth reduces computational overhead

2. **Flexibility**:
   - Easy to add/remove clients (just add/remove heads)
   - Per-client heads allow specialization

3. **Robustness**:
   - Multiple feature extraction paths provide redundancy
   - Batch normalization stabilizes training across heterogeneous data

4. **Federated Learning Compatibility**:
   - Architecture aligns with federated learning principles
   - Supports asynchronous updates and staleness handling

### Trade-offs

1. **Parameter Count**:
   - More parameters than single-branch network
   - Multiple heads increase model size linearly with number of clients

2. **Memory Usage**:
   - Each client stores full model (including all heads)
   - Aggregation requires storing updates for all parameters

3. **Communication Overhead**:
   - Model updates include changes to all heads
   - Larger models mean more data to transmit

4. **Scalability**:
   - Adding many clients increases model size
   - May need head pruning or sharing strategies for very large systems

## Code Examples

### Creating a Multihead Model

```python
from core.models import EnhancedMultiHeadDNN
import torch

# Initialize model for 5 clients with 128-dimensional input
model = EnhancedMultiHeadDNN(
    input_size=128,
    num_clients=5,
    dropout_rate=0.3
)

# Move to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

### Training Example

```python
# Prepare data
X_train = torch.randn(1000, 128)  # 1000 samples, 128 features
Y_train = torch.randn(1000, 5)    # 1000 samples, 5 clients (one-hot or probabilities)

# Forward pass
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

for epoch in range(10):
    optimizer.zero_grad()
    predictions = model(X_train)  # Shape: (1000, 5)
    loss = criterion(predictions, Y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

### Inference Example

```python
# Inference mode
model.eval()
with torch.no_grad():
    query = torch.randn(1, 128)  # Single query
    probs = model(query)  # Shape: (1, 5)
    predicted_client = probs.argmax(dim=1).item()
    confidence = probs.max().item()
    
    print(f"Predicted client: {predicted_client}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Probability distribution: {probs.squeeze().tolist()}")
```

### Integration in Federated Learning

```python
from core.federated_learning import EnhancedBlockchainFederatedLearning
import torch

# Initialize federated learning system
fl_system = EnhancedBlockchainFederatedLearning(
    num_clients=5,
    committee_size=3,
    input_size=128,
    num_outputs=5,  # Number of heads = number of clients
    device=torch.device('cpu'),
    enable_async=True
)

# Initialize clients with data
client_data_list = [
    (X_train_0, Y_train_0),  # Client 0 data
    (X_train_1, Y_train_1),  # Client 1 data
    # ... etc
]
fl_system.initialize_clients(client_data_list)

# Run federated round
results = fl_system.federated_round_async(
    local_epochs=1,
    batch_size=128,
    lr=0.001
)
```

## Summary

The **EnhancedMultiHeadDNN** architecture is specifically designed for federated learning scenarios where:

1. **Multiple clients** contribute to a shared model
2. **Per-client predictions** are needed (client identification, similarity matching)
3. **Efficient training** is required (shallow depth, parallel processing)
4. **Robustness** is important (regularization, stable aggregation)

The multihead design enables the model to:
- Learn shared feature representations across clients
- Specialize predictions for each client through dedicated heads
- Support efficient federated aggregation
- Scale to different numbers of clients

This architecture is well-suited for the blockchain-based federated learning system, where security, transparency, and efficiency are paramount.

## Related Documentation

- [Architecture Overview](architecture/ARCHITECTURE.md)
- [System Design](architecture/SYSTEM_DESIGN.md)
- [API Reference](api/API_REFERENCE.md)
- [Training Explanation](TRAINING_EXPLANATION.md)

