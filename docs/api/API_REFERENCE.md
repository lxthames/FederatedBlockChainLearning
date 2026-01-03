# API Reference

Complete API documentation for all classes and functions in the Blockchain-Based Decentralized Federated Learning Framework.

## Table of Contents

- [Model Classes](#model-classes)
- [Blockchain Classes](#blockchain-classes)
- [Security Classes](#security-classes)
- [Federated Learning Classes](#federated-learning-classes)
- [Utility Classes](#utility-classes)
- [Main Functions](#main-functions)

## Model Classes

### EnhancedMultiHeadDNN

Deep neural network with parallel branches and multi-head outputs.

```python
class EnhancedMultiHeadDNN(nn.Module)
```

#### Constructor

```python
__init__(self, input_size: int, num_clients: int, dropout_rate: float = 0.3)
```

**Parameters:**
- `input_size` (int): Size of input features
- `num_clients` (int): Number of clients (determines number of heads)
- `dropout_rate` (float): Dropout probability (default: 0.3)

**Returns:** None

#### Methods

##### forward

```python
forward(self, x: torch.Tensor) -> torch.Tensor
```

Forward pass through the network.

**Parameters:**
- `x` (torch.Tensor): Input tensor of shape `(batch_size, input_size)`

**Returns:**
- `torch.Tensor`: Output tensor of shape `(batch_size, num_clients)`

**Process:**
1. Extract features through two parallel branches
2. Merge features by averaging
3. Pass through multi-head outputs
4. Apply softmax normalization

##### _initialize_weights

```python
_initialize_weights(self) -> None
```

Initialize network weights using Kaiming initialization.

## Blockchain Classes

### Block

Represents a single block in the blockchain.

```python
class Block
```

#### Constructor

```python
__init__(self, index: int, transactions: List[Dict], previous_hash: str, 
         committee_votes: Dict[int, bool] = None)
```

**Parameters:**
- `index` (int): Block index
- `transactions` (List[Dict]): List of transactions
- `previous_hash` (str): Hash of previous block
- `committee_votes` (Dict[int, bool]): Committee member votes

#### Methods

##### calculate_hash

```python
calculate_hash(self) -> str
```

Calculate SHA-256 hash of the block.

**Returns:** `str` - Block hash

##### to_dict

```python
to_dict(self) -> Dict
```

Convert block to dictionary for serialization.

**Returns:** `Dict` - Block as dictionary

### Blockchain

Base blockchain implementation.

```python
class Blockchain
```

#### Constructor

```python
__init__(self)
```

Creates genesis block and initializes chain.

#### Methods

##### create_genesis_block

```python
create_genesis_block(self) -> Block
```

Create the first block in the chain.

**Returns:** `Block` - Genesis block

##### get_latest_block

```python
get_latest_block(self) -> Block
```

Get the most recent block.

**Returns:** `Block` - Latest block

##### add_transaction

```python
add_transaction(self, client_id: int, model_update: Dict, 
                local_epochs: int, data_size: int, loss: float) -> None
```

Add a transaction to pending transactions.

**Parameters:**
- `client_id` (int): Client identifier
- `model_update` (Dict): Serialized model update
- `local_epochs` (int): Number of local training epochs
- `data_size` (int): Size of training data
- `loss` (float): Training loss

##### mine_block

```python
mine_block(self, committee_votes: Dict[int, bool]) -> Block
```

Create a new block with pending transactions.

**Parameters:**
- `committee_votes` (Dict[int, bool]): Committee voting results

**Returns:** `Block` - Newly created block

##### validate_chain

```python
validate_chain(self) -> bool
```

Validate the integrity of the blockchain.

**Returns:** `bool` - True if chain is valid

### SignedBlockchain

Blockchain with signature support and persistence.

```python
class SignedBlockchain(Blockchain)
```

#### Constructor

```python
__init__(self, persistence: ChainPersistence = None)
```

**Parameters:**
- `persistence` (ChainPersistence): Persistence handler

#### Methods

##### add_transaction

```python
add_transaction(self, client_id: int, model_update: Dict, 
                local_epochs: int, data_size: int, loss: float,
                signature: str, public_id: str) -> None
```

Add a signed transaction.

**Additional Parameters:**
- `signature` (str): HMAC signature
- `public_id` (str): Public client identifier

## Security Classes

### ProofOfPersonhood

Proof-of-Personhood mechanism to prevent Sybil attacks.

```python
class ProofOfPersonhood
```

#### Constructor

```python
__init__(self, min_trust_score: float = 0.5)
```

**Parameters:**
- `min_trust_score` (float): Minimum trust score to participate (default: 0.5)

#### Methods

##### register_node

```python
register_node(self, node_id: int, verification_method: str = "challenge") -> Tuple[bool, str]
```

Register a new node with PoP verification.

**Parameters:**
- `node_id` (int): Node identifier
- `verification_method` (str): "challenge" or "web_of_trust"

**Returns:**
- `Tuple[bool, str]`: (success, token_or_error)

##### verify_node

```python
verify_node(self, node_id: int) -> bool
```

Verify if a node is registered and verified.

**Parameters:**
- `node_id` (int): Node identifier

**Returns:** `bool` - True if verified

##### can_participate

```python
can_participate(self, node_id: int) -> bool
```

Check if node can participate (registered + trust score threshold).

**Parameters:**
- `node_id` (int): Node identifier

**Returns:** `bool` - True if can participate

##### get_trust_score

```python
get_trust_score(self, node_id: int) -> float
```

Get trust score of a node.

**Parameters:**
- `node_id` (int): Node identifier

**Returns:** `float` - Trust score (0.0 to 1.0)

##### endorse_node

```python
endorse_node(self, endorser_id: int, node_id: int) -> bool
```

Endorse a node in web-of-trust.

**Parameters:**
- `endorser_id` (int): Endorser identifier
- `node_id` (int): Node to endorse

**Returns:** `bool` - True if successful

### ProofOfContribution

Proof-of-Contribution mechanism to evaluate update utility.

```python
class ProofOfContribution
```

#### Constructor

```python
__init__(self, validation_data: Optional[Tuple] = None)
```

**Parameters:**
- `validation_data` (Optional[Tuple]): (X_val, Y_val) for evaluation

#### Methods

##### evaluate_contribution

```python
evaluate_contribution(self, client_id: int, model_update: Dict, 
                     local_model: nn.Module, global_model: nn.Module,
                     validation_data: Optional[Tuple] = None,
                     device: torch.device = None) -> Dict[str, float]
```

Evaluate contribution of an update.

**Parameters:**
- `client_id` (int): Client identifier
- `model_update` (Dict): Model update dictionary
- `local_model` (nn.Module): Local model with update
- `global_model` (nn.Module): Current global model
- `validation_data` (Optional[Tuple]): Validation data
- `device` (torch.device): Computation device

**Returns:**
- `Dict[str, float]`: Contribution metrics
  - `accuracy_improvement`: Accuracy improvement score
  - `loss_reduction`: Loss reduction score
  - `robustness_score`: Robustness score
  - `overall_score`: Weighted overall score

##### rank_updates

```python
rank_updates(self, contributions: Dict[int, Dict[str, float]], 
             top_k: Optional[int] = None) -> List[Tuple[int, float]]
```

Rank updates by contribution score.

**Parameters:**
- `contributions` (Dict[int, Dict[str, float]]): Contribution scores per client
- `top_k` (Optional[int]): Return only top K updates

**Returns:**
- `List[Tuple[int, float]]`: List of (client_id, score) sorted descending

##### meets_threshold

```python
meets_threshold(self, contribution_score: float, threshold: float = 0.1) -> bool
```

Check if contribution meets minimum threshold.

**Parameters:**
- `contribution_score` (float): Overall contribution score
- `threshold` (float): Minimum threshold (default: 0.1)

**Returns:** `bool` - True if meets threshold

##### get_average_contribution

```python
get_average_contribution(self, client_id: int) -> float
```

Get average contribution score for a client.

**Parameters:**
- `client_id` (int): Client identifier

**Returns:** `float` - Average contribution score

### IdentityManager

Manages client identities and signatures.

```python
class IdentityManager
```

#### Constructor

```python
__init__(self, num_clients: int)
```

**Parameters:**
- `num_clients` (int): Number of clients

#### Methods

##### sign

```python
sign(self, client_id: int, payload: Dict) -> str
```

Sign a payload with client's secret key.

**Parameters:**
- `client_id` (int): Client identifier
- `payload` (Dict): Data to sign

**Returns:** `str` - HMAC signature

##### verify

```python
verify(self, client_id: int, payload: Dict, signature: str) -> bool
```

Verify a signature.

**Parameters:**
- `client_id` (int): Client identifier
- `payload` (Dict): Original data
- `signature` (str): Signature to verify

**Returns:** `bool` - True if signature is valid

## Federated Learning Classes

### FederatedClient

Represents a federated learning client.

```python
class FederatedClient
```

#### Constructor

```python
__init__(self, client_id: int, model: nn.Module, 
         data: Tuple, device: torch.device)
```

**Parameters:**
- `client_id` (int): Client identifier
- `model` (nn.Module): Local model instance
- `data` (Tuple): (X_train, Y_train) training data
- `device` (torch.device): Computation device

#### Methods

##### train_local

```python
train_local(self, epochs: int, batch_size: int, lr: float) -> float
```

Train model locally.

**Parameters:**
- `epochs` (int): Number of training epochs
- `batch_size` (int): Batch size
- `lr` (float): Learning rate

**Returns:** `float` - Final training loss

##### get_model_update

```python
get_model_update(self, global_state: Dict) -> Dict
```

Compute model update (difference from global).

**Parameters:**
- `global_state` (Dict): Global model state dictionary

**Returns:** `Dict` - Model update dictionary

##### apply_global_update

```python
apply_global_update(self, global_state: Dict) -> None
```

Apply global model update to local model.

**Parameters:**
- `global_state` (Dict): Global model state dictionary

### EnhancedBlockchainFederatedLearning

Main orchestration class for blockchain federated learning.

```python
class EnhancedBlockchainFederatedLearning
```

#### Constructor

```python
__init__(self, num_clients: int, committee_size: int, input_size: int,
         num_outputs: int, device: torch.device,
         election_method: str = "score", enable_async: bool = True,
         persistence_path: str = "blockchain_state.json",
         validation_data: Optional[Tuple] = None,
         poc_threshold: float = 0.1,
         top_k_updates: Optional[int] = None)
```

**Parameters:**
- `num_clients` (int): Number of clients
- `committee_size` (int): Committee size
- `input_size` (int): Input feature size
- `num_outputs` (int): Number of output classes
- `device` (torch.device): Computation device
- `election_method` (str): "score", "random", or "multi_factor"
- `enable_async` (bool): Enable asynchronous training
- `persistence_path` (str): Path for blockchain persistence
- `validation_data` (Optional[Tuple]): Validation data for PoC
- `poc_threshold` (float): Minimum PoC score threshold
- `top_k_updates` (Optional[int]): Top-K updates to accept

#### Methods

##### initialize_clients

```python
initialize_clients(self, client_data_list: List[Tuple]) -> None
```

Initialize clients and register with PoP.

**Parameters:**
- `client_data_list` (List[Tuple]): List of (X_train, Y_train) per client

##### federated_round_async

```python
federated_round_async(self, local_epochs: int = 1, batch_size: int = 128,
                     lr: float = 0.001, max_staleness: int = 3) -> Dict
```

Execute one federated learning round.

**Parameters:**
- `local_epochs` (int): Local training epochs
- `batch_size` (int): Batch size
- `lr` (float): Learning rate
- `max_staleness` (int): Maximum allowed staleness

**Returns:**
- `Dict`: Round statistics
  - `validated_updates`: Number of validated updates
  - `total_updates`: Total number of updates
  - `block_index`: Current block index
  - `consensus_achieved`: Whether consensus was achieved
  - `current_version`: Current model version

##### aggregate_updates_async

```python
aggregate_updates_async(self, validated_updates: List[Tuple]) -> None
```

Aggregate validated updates into global model.

**Parameters:**
- `validated_updates` (List[Tuple]): List of (update_info, metrics, score)

##### start_new_round

```python
start_new_round(self) -> None
```

Start a new training round and re-elect committee.

##### get_blockchain_info

```python
get_blockchain_info(self) -> Dict
```

Get blockchain information.

**Returns:**
- `Dict`: Blockchain statistics

##### get_client_statistics

```python
get_client_statistics(self) -> Dict[int, Dict]
```

Get statistics for all clients.

**Returns:**
- `Dict[int, Dict]`: Client statistics per client ID

## Utility Classes

### ModelVersion

Manages model versioning.

```python
class ModelVersion
```

#### Methods

##### increment_version

```python
increment_version(self) -> int
```

Increment and return new version number.

##### get_current_version

```python
get_current_version(self) -> int
```

Get current model version.

##### store_version

```python
store_version(self, version: int, state_dict: Dict) -> None
```

Store model state for a version.

### ChainPersistence

Handles blockchain persistence.

```python
class ChainPersistence
```

#### Methods

##### save

```python
save(self, chain: List[Dict]) -> None
```

Save chain to disk.

##### load

```python
load(self) -> Optional[List[Dict]]
```

Load chain from disk.

## Main Functions

### main_blockchain_federated

Main function for blockchain federated learning.

```python
def main_blockchain_federated() -> None
```

Executes complete federated learning training with blockchain.

**Process:**
1. Load and prepare data
2. Initialize federated learning system
3. Run federated rounds
4. Save results

### main_centralized

Original centralized training (for comparison).

```python
def main_centralized() -> None
```

## Related Documentation

- [Architecture Overview](../architecture/ARCHITECTURE.md)
- [Usage Guide](../guides/USAGE_GUIDE.md)
- [Setup Guide](../guides/SETUP_GUIDE.md)

