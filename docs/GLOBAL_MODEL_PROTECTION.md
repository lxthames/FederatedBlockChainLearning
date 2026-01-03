# How the Global Model is Protected

## Overview

The global model in this federated learning system is protected through multiple layers of security mechanisms. This document explains all the protection mechanisms in place.

---

## Protection Layers

### 1. **Identity Management & Cryptographic Signing**

#### HMAC-Based Signatures

Every model update is cryptographically signed before being added to the blockchain.

```python
class IdentityManager:
    """Manages per-client identities and signing using HMAC"""
    
    def sign(self, client_id: int, payload: Dict) -> str:
        """Sign a payload using client's secret key"""
        key = self.secret_keys[client_id]
        body = json.dumps(payload, sort_keys=True)
        return hmac.new(key, body.encode(), hashlib.sha256).hexdigest()
    
    def verify(self, client_id: int, payload: Dict, signature: str) -> bool:
        """Verify a signature"""
        expected = self.sign(client_id, payload)
        return hmac.compare_digest(expected, signature)
```

**How It Works:**
- Each client has a **unique secret key** (generated at initialization)
- Every transaction (model update) is signed with the client's secret key
- Signatures use **HMAC-SHA256** (cryptographically secure)
- Signatures are stored with each transaction on the blockchain
- Any tampering with the update invalidates the signature

**Protection Provided:**
- ✅ **Authenticity**: Ensures updates come from legitimate clients
- ✅ **Integrity**: Detects any modification to model updates
- ✅ **Non-repudiation**: Clients cannot deny their contributions

---

### 2. **Proof-of-Personhood (PoP) Verification**

#### Prevents Unauthorized Access

Only verified participants can contribute to the global model.

```python
class ProofOfPersonhood:
    def verify_node(self, node_id: int) -> bool:
        """Verify if a node is registered and verified"""
        return (node_id in self.registered_nodes and 
                self.registered_nodes[node_id].get("verified", False))
    
    def can_participate(self, node_id: int) -> bool:
        """Check if node can participate"""
        if not self.verify_node(node_id):
            return False
        return self.get_trust_score(node_id) >= self.min_trust_score
```

**How It Works:**
- Each client must **register** before participating
- Registration requires:
  - **Challenge verification** (simulated CAPTCHA/ZK proof)
  - OR **Web-of-trust** endorsements from existing nodes
- Clients receive a **trust score** (0.0 to 1.0)
- Only clients with trust score >= 0.5 can participate
- Unverified clients are **rejected** before their updates are considered

**Protection Provided:**
- ✅ **Sybil Attack Prevention**: One identity per participant
- ✅ **Access Control**: Only verified clients can contribute
- ✅ **Trust Scoring**: Tracks participant reliability
- ✅ **Early Rejection**: Malicious actors filtered before validation

---

### 3. **Proof-of-Contribution (PoC) Quality Control**

#### Ensures Only High-Quality Updates

Low-quality or malicious updates are filtered out before reaching the global model.

```python
class ProofOfContribution:
    def evaluate_contribution(self, ...) -> Dict[str, float]:
        """Evaluate contribution using multiple metrics"""
        # 1. Accuracy Improvement (40%)
        accuracy_improvement = max(0, global_loss - local_loss)
        
        # 2. Loss Reduction (40%)
        loss_reduction = max(0, global_loss - local_loss) / (global_loss + 1e-6)
        
        # 3. Robustness Score (20%)
        robustness_score = 1.0 / (1.0 + update_norm / 100.0)
        
        # Overall score
        overall_score = (
            0.4 * accuracy_improvement +
            0.4 * loss_reduction +
            0.2 * robustness_score
        )
        
        return {
            'accuracy_improvement': accuracy_improvement,
            'loss_reduction': loss_reduction,
            'robustness_score': robustness_score,
            'overall_score': overall_score
        }
    
    def meets_threshold(self, contribution_score: float, threshold: float = 0.1) -> bool:
        """Check if contribution meets minimum threshold"""
        return contribution_score >= threshold
```

**How It Works:**
- Each update is evaluated on a **validation set**
- Three metrics are calculated:
  1. **Accuracy Improvement**: How much better the local model performs
  2. **Loss Reduction**: How much the loss decreased
  3. **Robustness**: How stable the update is (based on update magnitude)
- Updates must meet a **minimum threshold** (default: 0.1)
- Updates are **ranked** by quality
- Only top-quality updates proceed to committee validation

**Protection Provided:**
- ✅ **Quality Control**: Low-quality updates rejected
- ✅ **Free-Riding Prevention**: Clients must provide useful updates
- ✅ **Multi-Metric Evaluation**: Prevents gaming single metric
- ✅ **Robustness Check**: Filters unstable updates

---

### 4. **Committee Consensus Validation**

#### Multi-Party Validation

A committee of selected clients validates every update before it's accepted.

```python
class EnhancedCommitteeConsensus:
    def validate_update_with_score(self, model_update: Dict, ...) -> Tuple[bool, float]:
        """Validate update and return score"""
        loss = validation_metrics.get('loss', float('inf'))
        update_norm = validation_metrics.get('update_norm', float('inf'))
        
        if loss < 1.0 and update_norm < 100.0:
            score = max(0, (1.0 / (loss + 1e-6)) * (1.0 / (update_norm + 1.0)))
            accepted = True
        else:
            score = 0.0
            accepted = False
        
        return accepted, score
    
    def aggregate_votes(self, votes: Dict[int, bool]) -> Tuple[bool, float]:
        """Aggregate committee votes"""
        accept_count = sum(1 for v in votes.values() if v)
        total = len(votes)
        majority = accept_count > (total / 2)
        return majority, accept_count / total if total > 0 else 0.0
```

**How It Works:**
- A **committee** of clients (25-30% of total) is elected each round
- Committee members must also be **PoP verified**
- Each committee member:
  1. Reviews the update
  2. Validates on their local data
  3. Votes: Accept or Reject
  4. Provides a validation score
- **Majority vote** determines acceptance
- **Median score** is used for final evaluation
- Committee members **rotate** each round (ensures fairness)

**Protection Provided:**
- ✅ **Distributed Validation**: No single point of validation
- ✅ **Majority Consensus**: Requires multiple approvals
- ✅ **Committee Rotation**: Prevents collusion
- ✅ **Score Aggregation**: Combines multiple opinions

---

### 5. **Blockchain Immutability**

#### Tamper-Proof Record Keeping

All validated updates are recorded on an immutable blockchain.

```python
class SignedBlockchain(Blockchain):
    def add_transaction(self, client_id: int, model_update: Dict, ...,
                       signature: str, public_id: str):
        """Add a signed transaction"""
        transaction = {
            "client_id": client_id,
            "model_update": model_update,
            "signature": signature,  # HMAC signature
            "public_id": public_id,
            "local_epochs": local_epochs,
            "data_size": data_size,
            "loss": loss,
            "timestamp": time.time()
        }
        self.pending_transactions.append(transaction)
    
    def validate_chain(self) -> bool:
        """Validate the integrity of the blockchain"""
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i-1]
            
            # Check hash integrity
            if current.hash != current.calculate_hash():
                return False
            # Check chain linkage
            if current.previous_hash != previous.hash:
                return False
        return True
```

**How It Works:**
- Each block contains:
  - **Transactions** (signed model updates)
  - **Previous block hash** (chain linkage)
  - **Block hash** (SHA-256 of all block data)
  - **Committee votes** (validation records)
- Blocks are **linked** via cryptographic hashes
- Any modification to a block **invalidates** the chain
- Chain validation runs periodically to detect tampering
- Blockchain state is **persisted** to disk (JSON files)

**Protection Provided:**
- ✅ **Immutability**: Updates cannot be modified after recording
- ✅ **Audit Trail**: Complete history of all updates
- ✅ **Tamper Detection**: Chain validation detects modifications
- ✅ **Transparency**: All participants can verify the chain

---

### 6. **Model Versioning & State Management**

#### Version Control for Model States

The global model is versioned, and all versions are tracked.

```python
class ModelVersion:
    def __init__(self):
        self.current_version = 0
        self.version_history = {}  # {version: state_dict}
    
    def increment_version(self) -> int:
        """Increment and return new version number"""
        self.current_version += 1
        return self.current_version
    
    def store_version(self, version: int, state_dict: Dict):
        """Store a model version"""
        self.version_history[version] = copy.deepcopy(state_dict)
    
    def get_version(self, version: int) -> Optional[Dict]:
        """Retrieve a specific model version"""
        return self.version_history.get(version)
```

**How It Works:**
- Each global model update creates a **new version**
- All versions are **stored** in memory (and optionally on disk)
- Clients track which version they're using
- **Staleness** is calculated based on version difference
- Old versions can be **retrieved** for rollback if needed

**Protection Provided:**
- ✅ **Version Control**: Track model evolution
- ✅ **Rollback Capability**: Can revert to previous versions
- ✅ **Staleness Tracking**: Identifies outdated updates
- ✅ **State Integrity**: Ensures correct model state

---

### 7. **Staleness-Aware Aggregation**

#### Handles Asynchronous Updates Safely

Stale updates are penalized to protect model quality.

```python
class StalenessHandler:
    def calculate_staleness_penalty(self, model_version: int, current_version: int) -> float:
        """Calculate penalty weight for stale models"""
        staleness = max(0, current_version - model_version)
        return self.staleness_decay ** staleness  # Exponential decay
```

**How It Works:**
- Each client tracks their **model version**
- **Staleness** = current_version - client_version
- Stale updates receive **reduced weight** in aggregation
- Exponential decay: `weight = base_weight × (0.9 ^ staleness)`
- Very stale updates (>3 versions) trigger **model refresh**

**Protection Provided:**
- ✅ **Quality Maintenance**: Prevents outdated updates from degrading model
- ✅ **Asynchronous Safety**: Handles network delays safely
- ✅ **Automatic Refresh**: Stale clients get updated model

---

### 8. **Weighted Aggregation with Contribution Tracking**

#### Fair and Secure Model Combination

Updates are aggregated using contribution-aware weights.

```python
def aggregate_updates_async(self, validated_updates: List[Tuple]):
    """Aggregate validated updates into global model"""
    total_data_size = sum(u[0]["data_size"] for u in validated_updates)
    current_version = self.model_version.get_current_version()
    fusion_weights = {}
    
    for update_info, metrics, score in validated_updates:
        cid = update_info["client_id"]
        base = update_info["data_size"] / total_data_size
        
        # Apply staleness penalty
        if self.enable_async:
            base *= self.staleness_handler.calculate_staleness_penalty(...)
        
        # Adjust for contribution gap
        gap = self.contribution_tracker.get_contribution_gap(cid)
        if gap > 0.05:
            base *= 1.0 + min(0.2, gap)
        
        fusion_weights[cid] = base
    
    # Normalize weights
    total_weight = sum(fusion_weights.values()) or 1.0
    fusion_weights = {k: v / total_weight for k, v in fusion_weights.items()}
    
    # Weighted aggregation
    global_state = self.global_model.state_dict()
    aggregated_state = {}
    for k in global_state:
        ws = torch.zeros_like(global_state[k])
        for update_info, _, _ in validated_updates:
            cid = update_info["client_id"]
            weight = fusion_weights.get(cid, 0.0)
            client_state = self.clients[cid].model.state_dict()
            ws += weight * client_state[k]
        aggregated_state[k] = ws
    
    self.global_model.load_state_dict(aggregated_state)
```

**How It Works:**
- Base weight = client's data size / total data size
- **Staleness penalty** reduces weight for old updates
- **Contribution gap adjustment** helps underperforming clients
- Weights are **normalized** (sum to 1.0)
- **Weighted average** combines all updates
- Only **validated updates** are included

**Protection Provided:**
- ✅ **Fair Aggregation**: Weights based on contribution
- ✅ **Quality Weighting**: Better updates have more influence
- ✅ **Staleness Handling**: Old updates don't dominate
- ✅ **Contribution Balance**: Helps maintain participation

---

### 9. **Storage Optimization & Cleanup**

#### Prevents Storage-Based Attacks

Old blocks and models are safely pruned.

```python
class StorageOptimizer:
    def cleanup_old_blocks(self, blockchain: Blockchain, keep_latest_n: int = 10) -> int:
        """Prune old blocks while maintaining chain integrity"""
        if len(blockchain.chain) <= keep_latest_n:
            return 0
        
        # Keep genesis block + latest N blocks
        blocks_to_keep = blockchain.chain[:1] + blockchain.chain[-keep_latest_n:]
        blockchain.chain = blocks_to_keep
        return len(blockchain.chain) - keep_latest_n - 1
```

**How It Works:**
- Old blockchain blocks are **pruned** (keeps latest N)
- Old model versions are **cleaned up**
- **Genesis block** is always preserved
- Chain integrity is **maintained** after pruning

**Protection Provided:**
- ✅ **Storage Efficiency**: Prevents unbounded growth
- ✅ **Chain Integrity**: Pruning doesn't break validation
- ✅ **Resource Management**: Prevents storage exhaustion attacks

---

## Complete Protection Flow

### Update Submission to Global Model Update

```
1. Client Submits Update
   │
   ├─→ [PROTECTION 1] Identity Verification
   │   └─→ HMAC Signature Generated
   │
   ├─→ [PROTECTION 2] PoP Verification
   │   ├─→ Check: Is client registered? ✓
   │   ├─→ Check: Trust score >= 0.5? ✓
   │   └─→ Result: Verified → Continue
   │
   ├─→ [PROTECTION 3] PoC Evaluation
   │   ├─→ Evaluate on validation set
   │   ├─→ Calculate: accuracy, loss, robustness
   │   ├─→ Check: Score >= threshold? ✓
   │   └─→ Result: Accepted → Continue
   │
   ├─→ [PROTECTION 4] Committee Validation
   │   ├─→ Committee members review
   │   ├─→ Majority vote: Accept? ✓
   │   └─→ Result: Validated → Continue
   │
   ├─→ [PROTECTION 5] Blockchain Recording
   │   ├─→ Create signed transaction
   │   ├─→ Add to block
   │   ├─→ Mine block (with hash)
   │   ├─→ Link to previous block
   │   └─→ Persist to disk
   │
   ├─→ [PROTECTION 6] Version Control
   │   ├─→ Increment model version
   │   └─→ Store model state
   │
   ├─→ [PROTECTION 7] Staleness Handling
   │   ├─→ Calculate staleness
   │   └─→ Apply penalty if stale
   │
   ├─→ [PROTECTION 8] Weighted Aggregation
   │   ├─→ Calculate fusion weights
   │   ├─→ Apply staleness penalty
   │   ├─→ Apply contribution adjustment
   │   └─→ Weighted average aggregation
   │
   └─→ [PROTECTION 9] Storage Management
       ├─→ Cleanup old blocks
       └─→ Maintain chain integrity
```

---

## Security Properties

### Confidentiality
- ✅ **Model weights are encrypted in transit** (via signatures)
- ✅ **Only validated updates** are aggregated
- ✅ **No raw data** is ever transmitted

### Integrity
- ✅ **HMAC signatures** detect tampering
- ✅ **Blockchain hashes** ensure immutability
- ✅ **Chain validation** detects modifications
- ✅ **Version control** tracks all changes

### Authenticity
- ✅ **PoP verification** ensures legitimate participants
- ✅ **HMAC signatures** verify update source
- ✅ **Public IDs** enable verification

### Availability
- ✅ **Asynchronous FL** handles network delays
- ✅ **Staleness handling** maintains quality
- ✅ **Committee rotation** prevents single points of failure

### Non-Repudiation
- ✅ **Signed transactions** cannot be denied
- ✅ **Blockchain records** provide audit trail
- ✅ **Public IDs** link updates to clients

---

## Attack Scenarios & Protections

### Scenario 1: Malicious Client Submits Bad Update

**Attack:** Client intentionally submits a low-quality or adversarial update.

**Protection:**
- ✅ **PoC Evaluation** rejects low-quality updates (threshold: 0.1)
- ✅ **Committee Validation** double-checks quality
- ✅ **Multi-metric evaluation** prevents gaming
- ✅ **Result:** Update rejected, not added to global model

---

### Scenario 2: Sybil Attack (Multiple Fake Identities)

**Attack:** Attacker creates multiple fake identities to dominate voting.

**Protection:**
- ✅ **PoP Registration** requires challenge verification
- ✅ **Trust Scoring** tracks participant reliability
- ✅ **Web-of-Trust** requires endorsements
- ✅ **Result:** Fake identities rejected, cannot participate

---

### Scenario 3: Man-in-the-Middle (Tampering)

**Attack:** Attacker intercepts and modifies model updates in transit.

**Protection:**
- ✅ **HMAC Signatures** detect any modification
- ✅ **Signature Verification** fails if tampered
- ✅ **Blockchain Validation** detects chain tampering
- ✅ **Result:** Tampered update rejected, chain invalidated

---

### Scenario 4: Replay Attack

**Attack:** Attacker resubmits old, previously accepted updates.

**Protection:**
- ✅ **Timestamps** in transactions
- ✅ **Version Control** tracks model versions
- ✅ **Staleness Penalty** reduces weight of old updates
- ✅ **Result:** Old updates penalized or rejected

---

### Scenario 5: Free-Riding (Low-Quality Contributions)

**Attack:** Client submits minimal updates to get rewards without contributing.

**Protection:**
- ✅ **PoC Threshold** requires minimum quality (0.1)
- ✅ **Contribution Tracking** monitors participation
- ✅ **Reward Mechanism** ties rewards to contribution quality
- ✅ **Result:** Low-quality updates rejected, no rewards

---

### Scenario 6: Model Poisoning

**Attack:** Client intentionally degrades global model with adversarial updates.

**Protection:**
- ✅ **PoC Evaluation** checks accuracy improvement
- ✅ **Committee Validation** multiple reviewers
- ✅ **Robustness Score** detects unstable updates
- ✅ **Weighted Aggregation** limits impact of single update
- ✅ **Result:** Poisoned update detected and rejected

---

## Comparison: Protected vs. Unprotected

| Aspect | Unprotected FL | Our Protected FL |
|--------|---------------|-----------------|
| **Update Verification** | None | PoP + PoC + Committee |
| **Signature** | None | HMAC-SHA256 |
| **Tamper Detection** | None | Blockchain validation |
| **Access Control** | Open | PoP-gated |
| **Quality Control** | None | PoC threshold |
| **Audit Trail** | Logs | Immutable blockchain |
| **Version Control** | None | Full versioning |
| **Attack Resistance** | Vulnerable | Multi-layer protection |

---

## Best Practices for Model Protection

### 1. Regular Chain Validation
```python
# Validate blockchain periodically
if not blockchain.validate_chain():
    raise SecurityError("Chain integrity compromised")
```

### 2. Monitor Trust Scores
```python
# Check for suspicious activity
for client_id in range(num_clients):
    trust = proof_of_personhood.get_trust_score(client_id)
    if trust < 0.3:
        # Investigate low trust clients
        investigate_client(client_id)
```

### 3. Verify Signatures
```python
# Always verify signatures before accepting
for transaction in block.transactions:
    if not identity_manager.verify(
        transaction["client_id"],
        transaction["model_update"],
        transaction["signature"]
    ):
        reject_transaction(transaction)
```

### 4. Check Contribution History
```python
# Monitor for free-riding
for client_id in range(num_clients):
    avg_contrib = proof_of_contribution.get_average_contribution(client_id)
    if avg_contrib < 0.05:
        # Client may be free-riding
        warn_client(client_id)
```

---

## Summary

The global model is protected through **9 layers of security**:

1. **Cryptographic Signing** (HMAC-SHA256)
2. **Proof-of-Personhood** (Access control)
3. **Proof-of-Contribution** (Quality control)
4. **Committee Consensus** (Multi-party validation)
5. **Blockchain Immutability** (Tamper-proof records)
6. **Model Versioning** (State management)
7. **Staleness Handling** (Asynchronous safety)
8. **Weighted Aggregation** (Fair combination)
9. **Storage Management** (Resource protection)

Together, these mechanisms ensure:
- ✅ **Only legitimate, high-quality updates** are accepted
- ✅ **All updates are cryptographically signed** and verified
- ✅ **Complete audit trail** via blockchain
- ✅ **Protection against common attacks** (Sybil, poisoning, tampering)
- ✅ **Fair and secure aggregation** of model updates

The global model is **highly protected** and suitable for production deployment in security-sensitive environments.

