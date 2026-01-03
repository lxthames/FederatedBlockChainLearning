# Architectural Refactoring: Client-Local Validation & Committee Median Voting

## Overview

This document describes the major architectural changes to implement:
1. **Client-local validation sets** (remove global validation)
2. **Committee-based median voting** for PoC scoring
3. **Decentralized validation** where each client validates peer updates

## Changes Summary

### 1. Configuration Constants

Add these constants at the top of the notebook (after imports):

```python
# PoC Configuration Constants (from paper)
GAMMA_1 = 0.4  # Weight for median(u_loss) in PoC score
GAMMA_2 = 0.4  # Weight for median(u_acc) in PoC score
GAMMA_3 = 0.2  # Weight for u_rob (robustness) in PoC score
DELTA_TH = 0.1  # Threshold for update acceptance
LAMBDA = 0.01  # Lambda parameter for robustness calculation
VAL_SPLIT_RATIO = 0.2  # 20% of local data for validation
```

### 2. Modified FederatedClient Class

**Changes to `__init__` method:**
- Split local data into train/val sets
- Store `self.train_data` and `self.val_data` separately

**New method `validate_peer_update`:**
- Takes candidate_weights and global_weights
- Evaluates both on `self.val_data`
- Returns (u_loss, u_acc)

```python
class FederatedClient:
    """Represents a client in the federated learning system"""
    def __init__(self, client_id: int, model: nn.Module, local_data: Tuple, 
                 device: torch.device, val_split_ratio: float = VAL_SPLIT_RATIO):
        self.client_id = client_id
        self.model = model
        self.device = device
        self.local_epochs = 0
        
        # Split local data into train and validation sets
        X_full, Y_full = local_data
        n_samples = len(X_full)
        n_val = int(n_samples * val_split_ratio)
        
        # Random split
        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        self.train_data = (X_full[train_indices], Y_full[train_indices])
        self.val_data = (X_full[val_indices], Y_full[val_indices])
        self.data_size = len(self.train_data[0])
    
    def validate_peer_update(self, candidate_weights: Dict, global_weights: Dict, 
                            device: torch.device = None) -> Tuple[float, float]:
        """
        Validate a peer's update on this client's local validation set.
        
        Args:
            candidate_weights: Model weights with the update applied
            global_weights: Current global model weights
            device: Device to use for evaluation
            
        Returns:
            (u_loss, u_acc): Utility scores for loss and accuracy
        """
        if device is None:
            device = self.device
        
        X_val, Y_val = self.val_data
        criterion = nn.MSELoss()
        
        # Create temporary models for evaluation
        candidate_model = copy.deepcopy(self.model)
        global_model = copy.deepcopy(self.model)
        
        candidate_model.load_state_dict(candidate_weights)
        global_model.load_state_dict(global_weights)
        
        candidate_model.eval()
        global_model.eval()
        
        with torch.no_grad():
            X_val_tensor = torch.from_numpy(X_val).float().to(device)
            Y_val_tensor = torch.from_numpy(Y_val).float().to(device)
            
            # Evaluate candidate model
            candidate_preds = candidate_model(X_val_tensor)
            candidate_loss = criterion(candidate_preds, Y_val_tensor).item()
            
            # Calculate accuracy (1 - normalized MSE, higher is better)
            mse = criterion(candidate_preds, Y_val_tensor).item()
            candidate_acc = max(0, 1.0 - mse)  # Simple accuracy proxy
            
            # Evaluate global model
            global_preds = global_model(X_val_tensor)
            global_loss = criterion(global_preds, Y_val_tensor).item()
            global_mse = criterion(global_preds, Y_val_tensor).item()
            global_acc = max(0, 1.0 - global_mse)
            
            # Calculate utility scores
            # u_loss: improvement in loss (higher is better)
            u_loss = max(0, global_loss - candidate_loss) / (global_loss + 1e-6)
            
            # u_acc: improvement in accuracy (higher is better)
            u_acc = max(0, candidate_acc - global_acc)
        
        return (u_loss, u_acc)
    
    def train_local(self, epochs: int, batch_size: int = 128, lr: float = 0.001):
        """Train model locally on client's training data"""
        X_train, Y_train = self.train_data  # Use train_data instead of local_data
        # ... rest of the method remains the same ...
```

### 3. Remove Global Validation from ProofOfContribution

**Remove `validation_data` parameter from:**
- `ProofOfContribution.__init__`
- `ProofOfContribution.evaluate_contribution` (or make it optional/unused)
- `EnhancedBlockchainFederatedLearning.__init__`

**Simplified ProofOfContribution:**
```python
class ProofOfContribution:
    """Proof-of-Contribution mechanism - now uses committee voting"""
    def __init__(self):
        self.contribution_history = {}
        # No validation_data needed - clients have their own
    
    def calculate_robustness(self, model_update: Dict, lambda_param: float = LAMBDA) -> float:
        """
        Calculate robustness score based on update norm.
        This is a global calculation, not committee-based.
        """
        update_norm = sum(
            torch.norm(v).item() 
            for v in model_update.values() 
            if isinstance(v, torch.Tensor) and v.dtype.is_floating_point
        )
        # Robustness: smaller updates are more robust
        u_rob = 1.0 / (1.0 + lambda_param * update_norm)
        return u_rob
```

### 4. Refactored federated_round_async Method

**New flow:**
1. Local training (unchanged)
2. PoP verification (unchanged)
3. **Committee voting** (NEW):
   - For each update, committee members vote using their local validation sets
   - Collect u_loss and u_acc from each committee member
   - Calculate median of votes
4. Calculate PoC score using median formula
5. Apply threshold and continue

```python
def federated_round_async(self, local_epochs: int = 1, batch_size: int = 128,
                          lr: float = 0.001, max_staleness: int = 3) -> Dict:
    current_version = self.model_version.get_current_version()
    client_updates, client_metrics = [], []

    # Step 1: Local training (unchanged)
    for client in self.clients:
        # ... existing training code ...
        client_updates.append({...})
        client_metrics.append({...})

    # Step 2: PoP verification (unchanged)
    pop_verified_updates = []
    for update_info, metrics in zip(client_updates, client_metrics):
        client_id = update_info["client_id"]
        if self.proof_of_personhood.can_participate(client_id):
            pop_verified_updates.append((update_info, metrics))

    # Step 3: Committee-based PoC evaluation (NEW)
    poc_scores = {}
    committee_members = self.committee.committee_members
    
    for update_info, metrics in pop_verified_updates:
        client_id = update_info["client_id"]
        update = update_info["update"]
        
        # Get candidate model (local model with update)
        candidate_model = copy.deepcopy(self.clients[client_id].model)
        candidate_state = candidate_model.state_dict()
        # Apply update to candidate state
        for key in candidate_state:
            if key in update and isinstance(candidate_state[key], torch.Tensor):
                candidate_state[key] = candidate_state[key] + update[key]
        
        # Get global model state
        global_state = self.global_model.state_dict()
        
        # Collect votes from committee members
        votes_loss = []
        votes_acc = []
        
        for committee_member_id in committee_members:
            # Committee members must be PoP verified
            if not self.proof_of_personhood.can_participate(committee_member_id):
                continue
            
            # Committee member validates the update on their local validation set
            committee_client = self.clients[committee_member_id]
            u_loss, u_acc = committee_client.validate_peer_update(
                candidate_state, 
                global_state, 
                self.device
            )
            votes_loss.append(u_loss)
            votes_acc.append(u_acc)
        
        # Calculate median of votes (robust to malicious members)
        if votes_loss and votes_acc:
            median_u_loss = np.median(votes_loss)
            median_u_acc = np.median(votes_acc)
            
            # Calculate robustness (global, not committee-based)
            u_rob = self.proof_of_contribution.calculate_robustness(update, LAMBDA)
            
            # Final PoC score using median formula
            poc_score = (
                GAMMA_1 * median_u_loss +
                GAMMA_2 * median_u_acc +
                GAMMA_3 * u_rob
            )
            
            poc_scores[client_id] = {
                'u_loss': median_u_loss,
                'u_acc': median_u_acc,
                'u_rob': u_rob,
                'poc_score': poc_score,
                'votes_loss': votes_loss,
                'votes_acc': votes_acc
            }
        else:
            # No valid votes
            poc_scores[client_id] = {
                'u_loss': 0.0,
                'u_acc': 0.0,
                'u_rob': 0.0,
                'poc_score': 0.0,
                'votes_loss': [],
                'votes_acc': []
            }
    
    # Step 4: Filter by threshold
    accepted_updates = {
        cid: score for cid, score in poc_scores.items()
        if score['poc_score'] >= DELTA_TH
    }
    
    print(f"PoC: {len(accepted_updates)}/{len(pop_verified_updates)} updates passed threshold")
    
    # Step 5: Committee validation (optional additional layer)
    validated_updates = []
    validation_scores = {}
    
    for update_info, metrics in pop_verified_updates:
        client_id = update_info["client_id"]
        
        if client_id not in accepted_updates:
            continue
        
        # Additional committee validation (if needed)
        votes, scores = {}, []
        for member_id in committee_members:
            if not self.proof_of_personhood.can_participate(member_id):
                continue
            
            accepted, score = self.committee.validate_update_with_score(
                update_info["update"], metrics
            )
            votes[member_id] = accepted
            if accepted:
                scores.append(score)
        
        accepted, _ = self.committee.aggregate_votes(votes)
        if accepted and scores:
            median_score = np.median(scores)
            validation_scores[client_id] = median_score
            self.committee.record_validation_score(client_id, median_score)
            
            # Combine PoC score with committee validation score
            poc_score = poc_scores[client_id]['poc_score']
            combined_score = 0.6 * poc_score + 0.4 * median_score
            
            # ... rest of transaction creation ...
            validated_updates.append((update_info, metrics, combined_score))
    
    # ... rest of block mining and aggregation ...
```

### 5. Update initialize_clients Method

**Change:**
- Remove validation_data parameter
- Clients will split their own data

```python
def initialize_clients(self, client_data_list: List[Tuple]):
    """Initialize clients and register them with PoP"""
    self.clients = []
    total_data_size = 0
    
    for i, (X_train, Y_train) in enumerate(client_data_list):
        # Register with PoP
        success, token = self.proof_of_personhood.register_node(i, verification_method="challenge")
        if not success:
            print(f"⚠️ Client {i} PoP registration failed: {token}")
            continue
        
        client_model = copy.deepcopy(self.global_model)
        # Client will split data internally
        client = FederatedClient(i, client_model, (X_train, Y_train), self.device)
        self.clients.append(client)
        total_data_size += client.data_size  # Use client's train data size
        self.contribution_tracker.update_sample_share(i, client.data_size, total_data_size)
        print(f"✓ Client {i} registered with PoP (token: {token[:8]}...)")
    
    # ... rest remains the same ...
```

### 6. Update EnhancedBlockchainFederatedLearning.__init__

**Remove validation_data parameter:**
```python
def __init__(self, num_clients: int, committee_size: int, input_size: int,
             num_outputs: int, device: torch.device,
             election_method: str = "score", enable_async: bool = True,
             persistence_path: str = "blockchain_state.json",
             poc_threshold: float = DELTA_TH,  # Use constant
             top_k_updates: Optional[int] = None):
    # ... existing code ...
    
    # Remove: validation_data parameter
    # Remove: self.proof_of_contribution = ProofOfContribution(validation_data=validation_data)
    self.proof_of_contribution = ProofOfContribution()  # No validation_data needed
    self.poc_threshold = poc_threshold
```

## Implementation Checklist

- [ ] Add configuration constants (GAMMA_1, GAMMA_2, GAMMA_3, DELTA_TH, LAMBDA)
- [ ] Modify FederatedClient.__init__ to split data into train/val
- [ ] Add FederatedClient.validate_peer_update method
- [ ] Update FederatedClient.train_local to use self.train_data
- [ ] Remove validation_data from ProofOfContribution
- [ ] Add ProofOfContribution.calculate_robustness method
- [ ] Remove validation_data from EnhancedBlockchainFederatedLearning.__init__
- [ ] Refactor federated_round_async to use committee voting
- [ ] Update initialize_clients to not pass validation_data
- [ ] Update main_blockchain_federated to not create/use global validation set
- [ ] Test the new implementation

## Key Benefits

1. **Privacy**: Each client's validation data stays local
2. **Robustness**: Median aggregation resists malicious committee members
3. **Decentralization**: No global validation set needed
4. **Fairness**: Each client validates using their own data distribution

## Mathematical Formulation

### PoC Score Calculation

$$S_{PoC}(k) = \gamma_1 \cdot \text{Median}(u_{loss}) + \gamma_2 \cdot \text{Median}(u_{acc}) + \gamma_3 \cdot u_{rob}$$

Where:
- $u_{loss} = \frac{\max(0, L(w_{global}) - L(w_k))}{L(w_{global}) + \epsilon}$
- $u_{acc} = \max(0, Acc(w_k) - Acc(w_{global}))$
- $u_{rob} = \frac{1}{1 + \lambda \cdot ||\Delta w_k||}$

### Acceptance Criterion

Update $w_k$ is accepted if:
$$S_{PoC}(k) \geq \Delta_{TH}$$

## Testing

After implementation, verify:
1. Each client has local validation set
2. Committee members vote on updates
3. Median is used (not mean) for aggregation
4. No global validation set is accessed
5. PoC scores match the formula above

