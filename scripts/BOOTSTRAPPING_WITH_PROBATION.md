# Bootstrapping with Probation: Implementation Summary

## Overview

Implemented a **bootstrapping with probation** mechanism to solve the cold-start deadlock while maintaining security. This allows new nodes to participate initially, but mitigates risk through trust-weighted aggregation.

## Key Principles

1. **Bootstrapping**: New nodes get enough initial trust (0.5) to pass the first filter
2. **No Administrative Penalties**: Do NOT penalize trust for filter rejections
3. **Validation-Based Penalties**: Only penalize after update passes all filters and is validated as bad
4. **Trust-Weighted Aggregation**: Low-trust nodes contribute minimally until they prove themselves

---

## Implementation Details

### 1. Initial Trust

**Location**: `proof_of_personhood.py:59`

```python
"trust_score": 0.5,  # Bootstrapping: Initial trust allows first participation
"probation": True    # Mark as in probation period
```

**Rationale**:
- Initial trust = 0.5 (above threshold = 0.45)
- Allows new nodes to pass trust filter and submit updates
- Enables validation of contributions before penalizing

---

### 2. Trust Threshold Filter (Before PoC)

**Location**: `federated_learning.py:208-221`

```python
# Step 1.5: Trust threshold filter BEFORE PoC evaluation
# New nodes start with trust=0.5 (above threshold=0.45) to allow first participation
trust_filtered_updates = []
for update_info, metrics in pop_verified_updates:
    client_id = update_info["client_id"]
    trust_score = self.proof_of_personhood.get_trust_score(client_id)
    if trust_score >= self.trust_threshold:  # 0.45
        trust_filtered_updates.append((update_info, metrics))
    else:
        # No penalty for administrative rejection
        trust_rejected_early.append((client_id, trust_score))
```

**Key Points**:
- Trust check happens **BEFORE** PoC evaluation
- No trust penalty for administrative rejection
- Only updates with trust >= 0.45 proceed to PoC

---

### 3. No Penalties for Administrative Rejections

**Location**: `federated_learning.py:344-378`

**Principle**: Only penalize trust **AFTER** update passes all filters and is validated as bad.

**What is NOT penalized**:
- ❌ Trust threshold rejection (administrative)
- ❌ PoC threshold rejection (quality filter)
- ❌ Committee validation rejection (consensus filter)

**What IS penalized**:
- ✅ Update passes all filters but is malicious (validated bad update)
- ✅ Sybil update accepted → Strong penalty (-0.15)
- ✅ Honest update accepted → Reward (+0.02)

---

### 4. Trust-Weighted Aggregation

**Location**: `federated_learning.py:466-475`

```python
# Enhanced trust weighting: Quadratic scaling for stronger mitigation
trust_weight = max(0.0, min(1.0, trust_score ** 2))
base *= trust_weight
```

**Weighting Table**:
| Trust Score | Weight (Linear) | Weight (Quadratic) | Impact |
|-------------|----------------|-------------------|--------|
| 1.0 | 100% | 100% | Full contribution |
| 0.5 | 50% | 25% | Probation/new nodes |
| 0.4 | 40% | 16% | Low trust - minimal |
| 0.3 | 30% | 9% | Very low trust |
| 0.2 | 20% | 4% | Extremely low |

**Rationale**:
- Quadratic scaling provides stronger mitigation
- New nodes (trust=0.5) contribute 25% weight (reduced risk)
- Low-trust nodes (trust=0.4) contribute only 16% (minimal impact)

---

## Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: PoP Verification                                    │
│ - Check: Is client registered?                              │
│ - Initial trust = 0.5 ✓                                     │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 1.5: Trust Threshold Filter (NEW)                      │
│ - Check: trust >= 0.45?                                      │
│ - Initial trust = 0.5 ✓ (passes)                           │
│ - Low trust = 0.4 ✗ (rejected, NO penalty)                │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: PoC Evaluation                                      │
│ - Only processes trust-filtered updates                      │
│ - Rejected updates: NO penalty                             │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Committee Validation                                │
│ - Rejected updates: NO penalty                               │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Trust-Weighted Aggregation                          │
│ - trust=0.5 → 25% weight (probation)                        │
│ - trust=1.0 → 100% weight (proven)                           │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 5: Reputation Update                                   │
│ - Only penalize if update passed ALL filters and is bad     │
│ - Sybil accepted: -0.15 trust (strong penalty)              │
│ - Honest accepted: +0.02 trust (reward)                      │
│ - Probation removed after successful contribution           │
└─────────────────────────────────────────────────────────────┘
```

---

## Expected Behavior

### New Node (Honest)
1. **Round 1**: 
   - Trust = 0.5 (passes filter)
   - Weight = 25% (quadratic)
   - Update accepted → Trust = 0.52, Probation = False

2. **Round 2+**:
   - Trust = 0.52+ (above threshold)
   - Weight increases as trust increases
   - Eventually reaches trust = 1.0, weight = 100%

### New Node (Sybil)
1. **Round 1**:
   - Trust = 0.5 (passes filter)
   - Weight = 25% (reduced impact)
   - Malicious update accepted → Trust = 0.35 (strong penalty)

2. **Round 2**:
   - Trust = 0.35 (below threshold = 0.45)
   - **BLOCKED** before PoC (no penalty for rejection)

### Low-Trust Node (Previously Penalized)
1. **Any Round**:
   - Trust = 0.4 (below threshold)
   - **BLOCKED** before PoC
   - No penalty for administrative rejection
   - Can recover trust through good behavior (if mechanism exists)

---

## Benefits

1. **Solves Cold-Start**: New nodes can participate immediately
2. **Maintains Security**: Low-trust nodes contribute minimally (quadratic weighting)
3. **Fair Penalties**: Only penalize after validation, not for filter rejections
4. **Probation Period**: New nodes start in probation, removed after good contribution

---

## Configuration

- **Initial Trust**: 0.5 (configurable in `proof_of_personhood.py`)
- **Trust Threshold**: 0.45 (configurable in `federated_learning.py`)
- **Weighting Function**: Quadratic (`trust_score ** 2`)
- **Penalty for Validated Bad Update**: -0.15
- **Reward for Validated Good Update**: +0.02

---

## Testing Recommendations

1. **New Honest Node**: Verify trust increases from 0.5 → 0.52+ after first good update
2. **New Sybil Node**: Verify trust decreases from 0.5 → 0.35 after first bad update, then blocked
3. **Low-Trust Node**: Verify no penalty for administrative rejection
4. **Weight Verification**: Verify quadratic weighting (0.5 → 25%, 0.4 → 16%)

