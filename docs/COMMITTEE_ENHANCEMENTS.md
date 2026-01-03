# Committee Consensus Enhancements: Score Decay and Term Limits

## Overview

This document describes the enhancements to the `EnhancedCommitteeConsensus` class to add score decay and term limits, allowing new high-performing nodes to enter the committee.

## Problem

Without decay or term limits, historical scores ($\Sigma S_{PoC}^{(r)}$) can accumulate indefinitely, causing:
- Same high-performing nodes to dominate the committee
- New high-performing nodes unable to enter
- Reduced fairness and diversity in committee membership

## Solution

Two mechanisms have been added:

### 1. Score Decay Factor

Historical scores are multiplied by a decay factor each round:
```
node_scores[client_id] *= score_decay_factor
```

**Default**: `score_decay_factor = 0.95` (5% decay per round)

**Effect**: 
- Old scores fade over time
- Recent performance has more weight
- New high-performing nodes can catch up

### 2. Term Limits

Optional limit on consecutive rounds a node can serve in the committee.

**Default**: `term_limit = None` (no limit)

**Effect**:
- Prevents nodes from staying in committee indefinitely
- Forces rotation after specified rounds
- Ensures fairness and diversity

## Implementation

### Modified `__init__` Method

```python
def __init__(self, committee_size: int, total_clients: int, election_method: str = "score",
             score_decay_factor: float = 0.95, term_limit: int = None):
    """
    Args:
        committee_size: Number of committee members
        total_clients: Total number of clients
        election_method: "random", "score", or "multi_factor"
        score_decay_factor: Decay factor for historical scores (0.0-1.0)
                           Higher = slower decay. 0.95 means 5% decay per round.
        term_limit: Maximum consecutive rounds a node can be in committee (None = no limit)
    """
    super().__init__(committee_size, total_clients)
    self.election_method = election_method
    self.node_scores = {i: 0.0 for i in range(total_clients)}
    self.validation_scores = {}
    self.current_round = 0
    self.score_decay_factor = score_decay_factor  # NEW
    self.term_limit = term_limit  # NEW
    self.committee_history = {}  # NEW: tracks when nodes were in committee
    self.consecutive_terms = {i: 0 for i in range(total_clients)}  # NEW: consecutive rounds
```

### Modified `start_new_round` Method

```python
def start_new_round(self):
    """Start a new training round, apply score decay, and re-elect committee"""
    self.current_round += 1
    
    # Apply decay factor to historical scores
    for client_id in self.node_scores:
        self.node_scores[client_id] *= self.score_decay_factor
    
    # Update consecutive terms
    for client_id in range(self.total_clients):
        if client_id in self.committee_members:
            self.consecutive_terms[client_id] += 1
            if client_id not in self.committee_history:
                self.committee_history[client_id] = []
            self.committee_history[client_id].append(self.current_round - 1)
        else:
            self.consecutive_terms[client_id] = 0
    
    # Re-elect committee
    self.committee_members = self.elect_committee()
```

### Modified `elect_committee` Method

```python
def elect_committee(self, method: str = None) -> List[int]:
    """Elect committee using specified method with term limits"""
    method = method or self.election_method
    
    # ... existing election logic ...
    
    # Apply term limits if enabled
    if self.term_limit is not None:
        committee = self._apply_term_limits(committee)
    
    return sorted(committee)
```

### New `_apply_term_limits` Method

```python
def _apply_term_limits(self, candidate_committee: List[int]) -> List[int]:
    """
    Apply term limits: exclude nodes that have exceeded consecutive term limit.
    Replace them with next best candidates.
    """
    final_committee = []
    excluded = set()
    
    # First pass: add candidates that haven't exceeded term limit
    for node_id in candidate_committee:
        if self.consecutive_terms[node_id] < self.term_limit:
            final_committee.append(node_id)
        else:
            excluded.add(node_id)
    
    # Second pass: fill remaining slots with next best candidates
    if len(final_committee) < self.committee_size:
        all_nodes = sorted(
            [(i, self.node_scores[i]) for i in range(self.total_clients) 
             if i not in final_committee and i not in excluded],
            key=lambda x: x[1],
            reverse=True
        )
        
        for node_id, _ in all_nodes:
            if len(final_committee) >= self.committee_size:
                break
            if self.consecutive_terms[node_id] < self.term_limit:
                final_committee.append(node_id)
    
    return sorted(final_committee)
```

## Usage

### Basic Usage (with decay, no term limit)

```python
committee = EnhancedCommitteeConsensus(
    committee_size=3,
    total_clients=10,
    election_method="score",
    score_decay_factor=0.95,  # 5% decay per round
    term_limit=None  # No term limit
)
```

### With Term Limits

```python
committee = EnhancedCommitteeConsensus(
    committee_size=3,
    total_clients=10,
    election_method="score",
    score_decay_factor=0.95,
    term_limit=5  # Maximum 5 consecutive rounds
)
```

### Adjusting Decay Rate

- **Higher decay (0.98-0.99)**: Slower decay, historical performance matters more
- **Lower decay (0.90-0.95)**: Faster decay, recent performance matters more
- **Very low decay (0.80-0.90)**: Very fast decay, only recent performance matters

## Benefits

1. **Fairness**: New high-performing nodes can enter the committee
2. **Diversity**: Prevents same nodes from dominating
3. **Adaptability**: System adapts to changing node performance
4. **Flexibility**: Configurable decay and term limits

## Mathematical Formulation

### Score Decay

At round $r$, the historical score is:
$$S^{(r)} = S^{(r-1)} \times \alpha$$

Where $\alpha$ is the decay factor (default: 0.95).

After $n$ rounds without new contributions:
$$S^{(r+n)} = S^{(r)} \times \alpha^n$$

### Term Limits

A node can serve at most $T$ consecutive rounds:
$$\text{consecutive\_terms}[i] < T$$

After $T$ rounds, the node is excluded from the next election.

## Example

### Without Decay/Term Limits
- Round 1: Nodes [2, 5, 8] elected (scores: 0.9, 0.8, 0.7)
- Round 2: Nodes [2, 5, 8] elected again (scores: 0.95, 0.85, 0.75)
- Round 3: Nodes [2, 5, 8] elected again (scores: 1.0, 0.9, 0.8)
- **Problem**: Same nodes dominate, new nodes never enter

### With Decay (0.95) and Term Limit (3)
- Round 1: Nodes [2, 5, 8] elected (scores: 0.9, 0.8, 0.7)
- Round 2: Scores decay to [0.855, 0.76, 0.665], Nodes [2, 5, 8] elected (terms: 1, 1, 1)
- Round 3: Scores decay to [0.812, 0.722, 0.632], Nodes [2, 5, 8] elected (terms: 2, 2, 2)
- Round 4: Scores decay to [0.771, 0.686, 0.600], Node 2 hits term limit, Nodes [5, 8, 1] elected
  - Node 1 (new high performer) enters committee
  - Node 2 excluded due to term limit
- **Benefit**: Rotation occurs, new nodes can enter

## Integration

To use these enhancements, update the initialization in `EnhancedBlockchainFederatedLearning`:

```python
self.committee = EnhancedCommitteeConsensus(
    committee_size, 
    num_clients, 
    election_method,
    score_decay_factor=0.95,  # Add this
    term_limit=5  # Add this (optional)
)
```

## Related Documentation

- [Architecture Documentation](architecture/ARCHITECTURE.md)
- [API Reference](api/API_REFERENCE.md)
- [System Design](architecture/SYSTEM_DESIGN.md)

