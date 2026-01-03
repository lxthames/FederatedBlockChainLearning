"""
Core Utilities

Versioning, staleness handling, contribution tracking, incentives, and storage optimization.
"""

from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn


class ModelVersion:
    """Manages model versions for staleness tracking"""
    
    def __init__(self):
        """Initialize version manager"""
        self.current_version = 0
        self.version_history = {}  # {version: model_state}
    
    def increment_version(self) -> int:
        """Increment and return new version number"""
        self.current_version += 1
        return self.current_version
    
    def get_current_version(self) -> int:
        """Get current version number"""
        return self.current_version
    
    def store_version(self, version: int, model_state: Dict):
        """
        Store model state for a specific version.
        
        Args:
            version: Version number
            model_state: Model state dictionary
        """
        self.version_history[version] = model_state
    
    def get_version(self, version: int) -> Optional[Dict]:
        """
        Get model state for a specific version.
        
        Args:
            version: Version number
            
        Returns:
            Model state dictionary or None if not found
        """
        return self.version_history.get(version)


class ClientContributionTracker:
    """Tracks contribution of each client to the global model"""
    
    def __init__(self, num_clients: int):
        """
        Initialize contribution tracker.
        
        Args:
            num_clients: Number of clients
        """
        self.num_clients = num_clients
        self.contribution_history = {i: [] for i in range(num_clients)}
        self.sample_shares = {i: 0.0 for i in range(num_clients)}
        self.total_contributions = {i: 0.0 for i in range(num_clients)}
    
    def update_sample_share(self, client_id: int, data_size: int, total_data_size: int):
        """
        Update the sample share for a client.
        
        Args:
            client_id: ID of the client
            data_size: Size of client's dataset
            total_data_size: Total size of all datasets
        """
        self.sample_shares[client_id] = data_size / total_data_size if total_data_size > 0 else 0.0
    
    def record_contribution(self, client_id: int, contribution: float):
        """
        Record contribution made by a client in a round.
        
        Args:
            client_id: ID of the client
            contribution: Contribution value
        """
        self.contribution_history[client_id].append(contribution)
        self.total_contributions[client_id] += contribution
    
    def get_contribution_gap(self, client_id: int) -> float:
        """
        Calculate gap between sample share and actual contribution.
        
        Args:
            client_id: ID of the client
            
        Returns:
            Gap value (absolute difference)
        """
        sample_share = self.sample_shares.get(client_id, 0.0)
        total_contrib = self.total_contributions.get(client_id, 0.0)
        
        # Normalize contribution
        total_all = sum(self.total_contributions.values())
        actual_contrib_share = total_contrib / total_all if total_all > 0 else 0.0
        
        return abs(sample_share - actual_contrib_share)
    
    def get_client_stats(self, client_id: int) -> Dict:
        """
        Get statistics for a client.
        
        Args:
            client_id: ID of the client
            
        Returns:
            Dictionary with client statistics
        """
        return {
            "sample_share": self.sample_shares.get(client_id, 0.0),
            "total_contribution": self.total_contributions.get(client_id, 0.0),
            "contribution_gap": self.get_contribution_gap(client_id),
            "participation_count": len(self.contribution_history.get(client_id, []))
        }
    
    def get_all_contributions(self) -> Dict[int, float]:
        """Get all total contributions"""
        return self.total_contributions.copy()


class StalenessHandler:
    """Handles model staleness in asynchronous federated learning"""
    
    def __init__(self, staleness_decay: float = 0.9):
        """
        Initialize staleness handler.
        
        Args:
            staleness_decay: Decay factor for stale models (0.9 = 10% decay per version)
        """
        self.staleness_decay = staleness_decay
    
    def calculate_staleness_penalty(self, model_version: int, current_version: int) -> float:
        """
        Calculate penalty weight for stale models.
        
        Args:
            model_version: Version the model was trained on
            current_version: Current global model version
            
        Returns:
            Penalty weight (0.0 to 1.0)
        """
        staleness = max(0, current_version - model_version)
        return self.staleness_decay ** staleness
    
    def adjust_fusion_weight(self, base_weight: float, staleness: int, current_version: int) -> float:
        """
        Adjust fusion weight based on staleness.
        
        Args:
            base_weight: Base fusion weight
            staleness: Staleness value
            current_version: Current global model version
            
        Returns:
            Adjusted weight
        """
        penalty = self.calculate_staleness_penalty(
            current_version - staleness,
            current_version
        )
        return base_weight * penalty


class IncentiveMechanism:
    """Implements profit sharing by contribution incentive mechanism"""
    
    def __init__(self, base_reward: float = 1.0):
        """
        Initialize incentive mechanism.
        
        Args:
            base_reward: Base reward value
        """
        self.base_reward = base_reward
        self.reward_history = {}  # {client_id: [rewards]}
        self.total_rewards_distributed = 0.0
    
    def calculate_reward(self, client_id: int, validation_score: float, 
                        contribution: float) -> float:
        """
        Calculate reward based on validation score and contribution.
        
        Args:
            client_id: ID of the client
            validation_score: Validation score from committee
            contribution: Contribution value
            
        Returns:
            Calculated reward
        """
        # Reward = base_reward * validation_score * (1 + contribution)
        reward = self.base_reward * validation_score * (1.0 + contribution)
        return reward
    
    def distribute_rewards(self, client_rewards: Dict[int, float]):
        """
        Distribute rewards to clients.
        
        Args:
            client_rewards: Dictionary of {client_id: reward}
        """
        for client_id, reward in client_rewards.items():
            if client_id not in self.reward_history:
                self.reward_history[client_id] = []
            self.reward_history[client_id].append(reward)
            self.total_rewards_distributed += reward
    
    def get_client_total_reward(self, client_id: int) -> float:
        """
        Get total reward earned by a client.
        
        Args:
            client_id: ID of the client
            
        Returns:
            Total reward
        """
        return sum(self.reward_history.get(client_id, []))


class StorageOptimizer:
    """Optimizes blockchain storage by managing historical blocks"""
    
    def __init__(self, keep_latest_n_models: int = 10):
        """
        Initialize storage optimizer.
        
        Args:
            keep_latest_n_models: Number of latest models to keep
        """
        self.keep_latest_n_models = keep_latest_n_models
        self.model_storage = {}  # {version: model_state}
        self.update_storage = {}  # {block_index: updates}
    
    def store_model(self, version: int, model_state: Dict):
        """
        Store model and clean up old ones.
        
        Args:
            version: Version number
            model_state: Model state dictionary
        """
        self.model_storage[version] = model_state
        
        # Keep only latest N models
        if len(self.model_storage) > self.keep_latest_n_models:
            oldest_version = min(self.model_storage.keys())
            del self.model_storage[oldest_version]
    
    def get_latest_model(self) -> Optional[Dict]:
        """
        Get the latest stored model.
        
        Returns:
            Latest model state or None
        """
        if not self.model_storage:
            return None
        latest_version = max(self.model_storage.keys())
        return self.model_storage[latest_version]
    
    def cleanup_old_blocks(self, blockchain, keep_recent_blocks: int = 100) -> int:
        """
        Clean up old blocks (for nodes with limited storage).
        
        Args:
            blockchain: Blockchain instance
            keep_recent_blocks: Number of recent blocks to keep
            
        Returns:
            Number of blocks that can be pruned
        """
        if len(blockchain.chain) > keep_recent_blocks:
            return len(blockchain.chain) - keep_recent_blocks
        return 0

