"""
Second Layer of Security: Model Rollback + Staking & Slashing

Implements:
1. Model Rollback (Time Machine) - Reactive recovery from poisoning
2. Staking & Slashing (Skin in the Game) - Proactive economic security
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
import copy


# ============================================================================
# Configuration Constants
# ============================================================================
STAKE_REQUIRED = 10.0  # Minimum stake required to be committee candidate
PENALTY_AMOUNT = 5.0  # Amount slashed for approving poisoned model
ACCURACY_DROP_THRESHOLD = 0.05  # 5% drop triggers rollback
HISTORY_WINDOW = 5  # Number of rounds for moving average
ROLLBACK_DEPTH = 1  # How many blocks to revert (1 = previous block, 5 = 5 blocks back)


# ============================================================================
# Watchdog: Poisoning Detection
# ============================================================================
class PoisoningDetector:
    """
    Detects model poisoning by monitoring accuracy drops.
    Uses moving average to detect sudden degradation.
    """
    
    def __init__(self, accuracy_drop_threshold: float = ACCURACY_DROP_THRESHOLD,
                 history_window: int = HISTORY_WINDOW):
        """
        Args:
            accuracy_drop_threshold: Percentage drop that triggers alert (0.05 = 5%)
            history_window: Number of rounds for moving average
        """
        self.accuracy_drop_threshold = accuracy_drop_threshold
        self.history_window = history_window
        self.accuracy_history = deque(maxlen=history_window)
        self.poisoning_events = []  # List of detected events
    
    def detect_poisoning_event(self, current_accuracy: float) -> Tuple[bool, float]:
        """
        Detect if current accuracy indicates poisoning.
        
        Args:
            current_accuracy: Current test accuracy (0.0 to 1.0)
            
        Returns:
            (is_poisoned, drop_percentage): True if poisoning detected, and drop %
        """
        if len(self.accuracy_history) < self.history_window:
            # Not enough history yet
            self.accuracy_history.append(current_accuracy)
            return False, 0.0
        
        # Calculate moving average of recent accuracy
        moving_average = np.mean(list(self.accuracy_history))
        
        # Calculate drop percentage
        if moving_average > 0:
            drop_percentage = (moving_average - current_accuracy) / moving_average
        else:
            drop_percentage = 0.0
        
        # Check if drop exceeds threshold
        is_poisoned = drop_percentage > self.accuracy_drop_threshold
        
        if is_poisoned:
            self.poisoning_events.append({
                'round': len(self.accuracy_history),
                'current_accuracy': current_accuracy,
                'moving_average': moving_average,
                'drop_percentage': drop_percentage
            })
        
        # Update history
        self.accuracy_history.append(current_accuracy)
        
        return is_poisoned, drop_percentage
    
    def reset(self):
        """Reset detector state"""
        self.accuracy_history.clear()
        self.poisoning_events.clear()


# ============================================================================
# Staking System
# ============================================================================
class StakingSystem:
    """
    Manages staking and slashing for committee participation.
    """
    
    def __init__(self, stake_required: float = STAKE_REQUIRED,
                 penalty_amount: float = PENALTY_AMOUNT):
        """
        Args:
            stake_required: Minimum stake to be committee candidate
            penalty_amount: Amount slashed for bad behavior
        """
        self.stake_required = stake_required
        self.penalty_amount = penalty_amount
        self.stake_balances = {}  # {client_id: balance}
        self.locked_stakes = {}  # {client_id: locked_amount} - stakes locked in current round
        self.slashing_history = []  # History of slashing events
    
    def initialize_client(self, client_id: int, initial_stake: float = 0.0):
        """Initialize stake balance for a client"""
        self.stake_balances[client_id] = initial_stake
        self.locked_stakes[client_id] = 0.0
    
    def can_be_committee_member(self, client_id: int) -> bool:
        """Check if client has enough stake to be committee member"""
        available_stake = self.stake_balances.get(client_id, 0.0) - self.locked_stakes.get(client_id, 0.0)
        return available_stake >= self.stake_required
    
    def lock_stake(self, client_id: int, amount: float = None) -> bool:
        """
        Lock stake when client is elected to committee.
        
        Args:
            client_id: Client ID
            amount: Amount to lock (default: stake_required)
            
        Returns:
            True if successful, False if insufficient stake
        """
        if amount is None:
            amount = self.stake_required
        
        available = self.stake_balances.get(client_id, 0.0) - self.locked_stakes.get(client_id, 0.0)
        
        if available < amount:
            return False
        
        self.locked_stakes[client_id] = self.locked_stakes.get(client_id, 0.0) + amount
        return True
    
    def unlock_stake(self, client_id: int, amount: float = None):
        """Unlock stake after committee service"""
        if amount is None:
            amount = self.stake_required
        
        self.locked_stakes[client_id] = max(0.0, self.locked_stakes.get(client_id, 0.0) - amount)
    
    def slash_stake(self, client_id: int, reason: str = "Poisoning approval") -> float:
        """
        Slash (burn) stake for bad behavior.
        
        Args:
            client_id: Client ID to slash
            reason: Reason for slashing
            
        Returns:
            Amount slashed
        """
        current_balance = self.stake_balances.get(client_id, 0.0)
        locked = self.locked_stakes.get(client_id, 0.0)
        
        # Slash from both locked and available stake
        slash_amount = min(self.penalty_amount, current_balance)
        
        if slash_amount > 0:
            self.stake_balances[client_id] = max(0.0, current_balance - slash_amount)
            
            # Also reduce locked stake if applicable
            if locked > 0:
                self.locked_stakes[client_id] = max(0.0, locked - min(slash_amount, locked))
            
            self.slashing_history.append({
                'client_id': client_id,
                'amount': slash_amount,
                'reason': reason,
                'remaining_balance': self.stake_balances[client_id]
            })
        
        return slash_amount
    
    def add_stake(self, client_id: int, amount: float):
        """Add stake to client balance (e.g., as reward)"""
        self.stake_balances[client_id] = self.stake_balances.get(client_id, 0.0) + amount
    
    def get_stake_balance(self, client_id: int) -> float:
        """Get current stake balance"""
        return self.stake_balances.get(client_id, 0.0)
    
    def get_available_stake(self, client_id: int) -> float:
        """Get available stake (balance - locked)"""
        return self.stake_balances.get(client_id, 0.0) - self.locked_stakes.get(client_id, 0.0)


# ============================================================================
# Enhanced Blockchain with Rollback
# ============================================================================
class RollbackBlockchain:
    """
    Compatibility wrapper for application-layer model recovery.

    Confirmed blocks remain immutable. ``revert_to_block`` retains its legacy
    name but only restores a checkpoint and appends an in-memory recovery
    record; it never truncates the underlying chain.
    """
    
    def __init__(self, base_blockchain):
        """
        Args:
            base_blockchain: The existing SignedBlockchain instance
        """
        self.blockchain = base_blockchain
        self.model_checkpoints = {}  # {block_index: model_state_dict}
        self.rollback_history = []  # History of rollbacks
    
    def checkpoint_model(self, block_index: int, model_state: Dict):
        """Save model state at a specific block"""
        self.model_checkpoints[block_index] = copy.deepcopy(model_state)
    
    def revert_to_block(self, block_height: int) -> Dict:
        """
        Revert blockchain and return model state from target block.
        
        Args:
            block_height: Target block index to revert to
            
        Returns:
            Model state dictionary from that block
        """
        if block_height < 0 or block_height >= len(self.blockchain.chain):
            raise ValueError(f"Invalid block height: {block_height}")
        
        from_block = len(self.blockchain.chain) - 1
        
        # Get model state from checkpoint
        if block_height in self.model_checkpoints:
            model_state = copy.deepcopy(self.model_checkpoints[block_height])
        else:
            model_state = None
        
        self.rollback_history.append({
            'from_block': from_block,
            'to_block': block_height,
            'blocks_discarded': 0,
            'invalidated_versions': list(range(block_height + 1, from_block + 1)),
            'semantics': 'application_layer_recovery'
        })
        
        return model_state
    
    def get_block_committee(self, block_index: int) -> List[int]:
        """Get committee members who signed a specific block"""
        if block_index < 0 or block_index >= len(self.blockchain.chain):
            return []
        
        block = self.blockchain.chain[block_index]
        if block is not None and hasattr(block, 'committee_votes'):
            return [
                client_id
                for client_id, approved in block.committee_votes.items()
                if approved
            ]
        return []
    
    def get_latest_block_index(self) -> int:
        """Get index of latest block"""
        return len(self.blockchain.chain) - 1


# ============================================================================
# Security Layer Manager
# ============================================================================
class SecondLayerSecurity:
    """
    Manages the 2nd layer of security: Rollback + Staking & Slashing
    """
    
    def __init__(self, blockchain, staking_system: StakingSystem,
                 poisoning_detector: PoisoningDetector,
                 proof_of_personhood,
                 rollback_depth: int = ROLLBACK_DEPTH):
        """
        Args:
            blockchain: RollbackBlockchain instance
            staking_system: StakingSystem instance
            poisoning_detector: PoisoningDetector instance
            proof_of_personhood: PoP instance for trust score updates
            rollback_depth: How many blocks to revert
        """
        self.blockchain = blockchain
        self.staking_system = staking_system
        self.poisoning_detector = poisoning_detector
        self.proof_of_personhood = proof_of_personhood
        self.rollback_depth = rollback_depth
        self.security_events = []  # Log of security events
    
    def check_and_respond(self, current_accuracy: float, global_model) -> Tuple[bool, Optional[Dict]]:
        """
        Check for poisoning and respond if detected.
        
        Args:
            current_accuracy: Current test accuracy
            global_model: Current global model
            
        Returns:
            (poisoning_detected, model_state): True if rollback needed, and model state
        """
        is_poisoned, drop_percentage = self.poisoning_detector.detect_poisoning_event(current_accuracy)
        
        if not is_poisoned:
            return False, None
        
        # Poisoning detected! Trigger response
        print(f"\n⚠️ POISONING DETECTED!")
        print(f"   Accuracy drop: {drop_percentage*100:.2f}%")
        print(f"   Triggering rollback and slashing...")
        
        # Get current block index
        current_block = self.blockchain.get_latest_block_index()
        target_block = max(0, current_block - self.rollback_depth)
        
        # Identify bad committee members
        bad_committee = self.blockchain.get_block_committee(current_block)
        
        # Slash and ban bad committee members
        slashed_clients = []
        for client_id in bad_committee:
            # Slash stake
            slashed_amount = self.staking_system.slash_stake(client_id, "Poisoning approval")
            if slashed_amount > 0:
                slashed_clients.append((client_id, slashed_amount))
            
            # Ban: Set trust score to 0
            if hasattr(self.proof_of_personhood, 'registered_nodes'):
                if client_id in self.proof_of_personhood.registered_nodes:
                    self.proof_of_personhood.registered_nodes[client_id]['trust_score'] = 0.0
        
        # Revert blockchain
        model_state = self.blockchain.revert_to_block(target_block)
        
        # Log event
        self.security_events.append({
            'round': current_block,
            'type': 'poisoning_detected',
            'drop_percentage': drop_percentage,
            'target_block': target_block,
            'bad_committee': bad_committee,
            'slashed_clients': slashed_clients
        })
        
        print(f"   Rolled back to block {target_block}")
        print(f"   Slashed {len(slashed_clients)} committee members")
        print(f"   Banned {len(bad_committee)} clients (trust_score = 0)")
        
        return True, model_state
    
    def filter_committee_candidates(self, candidate_pool: List[int]) -> List[int]:
        """
        Filter committee candidates based on stake requirements.
        
        Args:
            candidate_pool: List of candidate client IDs
            
        Returns:
            Filtered list of clients with sufficient stake
        """
        eligible = [
            cid for cid in candidate_pool
            if self.staking_system.can_be_committee_member(cid)
        ]
        return eligible
    
    def lock_committee_stakes(self, committee_members: List[int]):
        """Lock stakes for committee members"""
        for member_id in committee_members:
            self.staking_system.lock_stake(member_id)
    
    def unlock_committee_stakes(self, committee_members: List[int]):
        """Unlock stakes after committee service"""
        for member_id in committee_members:
            self.staking_system.unlock_stake(member_id)
    
    def get_security_summary(self) -> Dict:
        """Get summary of security events"""
        return {
            'total_poisoning_events': len(self.poisoning_detector.poisoning_events),
            'total_rollbacks': len(self.blockchain.rollback_history),
            'total_slashings': len(self.staking_system.slashing_history),
            'security_events': self.security_events
        }

