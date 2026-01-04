"""
Committee Consensus

Manages committee selection, voting, and consensus mechanisms.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


class CommitteeConsensus:
    """Manages committee consensus for model update validation"""
    
    def __init__(self, committee_size: int, total_clients: int):
        """
        Initialize committee consensus.
        
        Args:
            committee_size: Number of committee members
            total_clients: Total number of clients in the system
        """
        self.committee_size = committee_size
        self.total_clients = total_clients
        self.committee_members = self.select_committee()
    
    def select_committee(self) -> List[int]:
        """
        Select committee members (random selection by default).
        
        Returns:
            List of committee member IDs
        """
        np.random.seed(42)
        return sorted(np.random.choice(
            self.total_clients, 
            size=min(self.committee_size, self.total_clients), 
            replace=False
        ).tolist())
    
    def vote_on_update(self, model_update: Dict, validation_metrics: Dict) -> bool:
        """
        Committee member votes on whether to accept a model update.
        
        Args:
            model_update: Model update dictionary
            validation_metrics: Validation metrics (loss, update_norm, etc.)
            
        Returns:
            True if update should be accepted, False otherwise
        """
        loss = validation_metrics.get('loss', float('inf'))
        update_norm = validation_metrics.get('update_norm', float('inf'))
        
        # Accept if loss is reasonable and update norm is not too large
        return loss < 1.0 and update_norm < 100.0
    
    def aggregate_votes(self, votes: Dict[int, bool]) -> Tuple[bool, float]:
        """
        Aggregate committee votes using 2/3 majority rule.
        
        Args:
            votes: Dictionary of {member_id: vote}
            
        Returns:
            (consensus_reached, consensus_ratio)
        """
        if not votes:
            return False, 0.0
        
        positive_votes = sum(1 for v in votes.values() if v)
        consensus_ratio = positive_votes / len(votes)
        threshold = 2.0 / 3.0
        
        return consensus_ratio >= threshold, consensus_ratio


class EnhancedCommitteeConsensus(CommitteeConsensus):
    """
    Enhanced committee consensus with scoring and multiple election methods.
    Includes score decay and term limits for fairness.
    """
    
    def __init__(self, committee_size: int, total_clients: int, 
                 election_method: str = "score",
                 score_decay_factor: float = 0.95,
                 term_limit: Optional[int] = None):
        """
        Initialize enhanced committee consensus.
        
        Args:
            committee_size: Number of committee members
            total_clients: Total number of clients
            election_method: Method for electing committee ("random", "score", "multi_factor")
            score_decay_factor: Decay factor for historical scores (0.95 = 5% decay per round)
            term_limit: Maximum consecutive rounds in committee (None = no limit)
        """
        super().__init__(committee_size, total_clients)
        self.election_method = election_method
        self.score_decay_factor = score_decay_factor
        self.term_limit = term_limit
        
        # Historical scores (decay over time)
        self.node_scores = {i: 0.0 for i in range(total_clients)}
        self.validation_scores = {}  # {round: {client_id: score}}
        self.current_round = 0
        self.committee_terms = {i: 0 for i in range(total_clients)}  # Consecutive rounds in committee
    
    def validate_update_with_score(self, model_update: Dict, validation_metrics: Dict, 
                                   validation_data: Tuple = None) -> Tuple[bool, float]:
        """
        Validate update and return score.
        
        Args:
            model_update: Model update dictionary
            validation_metrics: Validation metrics
            validation_data: Optional validation data tuple
            
        Returns:
            (accepted, score)
        """
        loss = validation_metrics.get('loss', float('inf'))
        update_norm = validation_metrics.get('update_norm', float('inf'))
        
        # Adaptive thresholds: more permissive for neural network updates
        # Update norms for DNNs are typically in thousands, not hundreds
        loss_threshold = 1.0  # Loss should be reasonable
        norm_threshold = 200000.0  # Allow larger norms for DNN weight updates
        
        # Calculate validation score (higher is better)
        if loss < loss_threshold and update_norm < norm_threshold:
            # Score: inverse of loss, penalized by update norm (normalized)
            # Normalize norm by dividing by 10000 to get reasonable scale
            normalized_norm = update_norm / 10000.0
            score = max(0, (1.0 / (loss + 1e-6)) * (1.0 / (normalized_norm + 1.0)))
            accepted = True
        else:
            score = 0.0
            accepted = False
        # #region agent log
        import json
        import time
        import os
        try:
            log_path = r'c:\Users\user\Desktop\FederatedBlockchainLearning\.cursor\debug.log'
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({"id": f"log_{int(time.time()*1000)}", "timestamp": int(time.time()*1000), "location": "consensus.py:131", "message": "Committee validation check", "data": {"loss": loss, "loss_threshold": loss_threshold, "update_norm": update_norm, "norm_threshold": norm_threshold, "accepted": accepted, "score": score}, "sessionId": "debug-session", "runId": "run1", "hypothesisId": "C"}) + '\n')
        except Exception as e:
            print(f"DEBUG LOG ERROR: {e}")
        # #endregion
        
        return accepted, score
    
    def record_validation_score(self, client_id: int, score: float):
        """
        Record validation score for a client and update historical score.
        
        Args:
            client_id: ID of the client
            score: Validation score
        """
        if self.current_round not in self.validation_scores:
            self.validation_scores[self.current_round] = {}
        self.validation_scores[self.current_round][client_id] = score
        
        # Update historical score (moving average with decay)
        self.node_scores[client_id] = (
            self.score_decay_factor * self.node_scores[client_id] + 
            (1 - self.score_decay_factor) * score
        )
    
    def elect_committee(self, method: str = None, exclude_clients: List[int] = None) -> List[int]:
        """
        Elect committee using specified method.
        
        Args:
            method: Election method ("random", "score", "multi_factor")
            exclude_clients: List of client IDs to exclude from committee (e.g., Sybils)
            
        Returns:
            List of elected committee member IDs
        """
        method = method or self.election_method
        exclude_clients = exclude_clients or []
        
        if method == "random":
            # Random selection excluding specified clients
            eligible = [i for i in range(self.total_clients) if i not in exclude_clients]
            if len(eligible) < self.committee_size:
                # Fallback: if not enough eligible, use all clients (with warning)
                eligible = list(range(self.total_clients))
                if len(eligible) < self.committee_size:
                    # Last resort: return all eligible clients (even if fewer than committee_size)
                    return sorted(eligible)
            np.random.seed(42 + self.current_round)  # Different seed per round
            return sorted(np.random.choice(eligible, size=min(self.committee_size, len(eligible)), replace=False).tolist())
        
        elif method == "score":
            # Select top performers based on scores, excluding specified clients
            sorted_nodes = sorted(
                [(node_id, score) for node_id, score in self.node_scores.items() if node_id not in exclude_clients], 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Apply term limits if enabled
            if self.term_limit is not None:
                # Filter out clients who have reached term limit
                eligible = [
                    (node_id, score) for node_id, score in sorted_nodes
                    if self.committee_terms[node_id] < self.term_limit
                ]
                if len(eligible) >= self.committee_size:
                    sorted_nodes = eligible
            
            # Ensure we have enough nodes
            if len(sorted_nodes) < self.committee_size:
                # Fallback: include excluded clients if necessary
                all_nodes = sorted(
                    [(node_id, score) for node_id, score in self.node_scores.items()], 
                    key=lambda x: x[1], 
                    reverse=True
                )
                sorted_nodes = all_nodes[:self.committee_size]
            
            committee = [node_id for node_id, _ in sorted_nodes[:self.committee_size]]
            return sorted(committee)
        
        elif method == "multi_factor":
            # Multi-factor optimization (score + participation)
            participation = {
                i: len([r for r in self.validation_scores.values() if i in r]) 
                for i in range(self.total_clients)
            }
            
            # Combined score: validation score + participation bonus
            combined_scores = {
                i: self.node_scores[i] + 0.1 * participation.get(i, 0)
                for i in range(self.total_clients)
            }
            
            # Apply term limits
            if self.term_limit is not None:
                combined_scores = {
                    i: score for i, score in combined_scores.items()
                    if self.committee_terms[i] < self.term_limit
                }
            
            sorted_nodes = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            committee = [node_id for node_id, _ in sorted_nodes[:self.committee_size]]
            return sorted(committee)
        
        else:
            return self.select_committee()
    
    def start_new_round(self):
        """Start a new training round, apply score decay, and re-elect committee"""
        self.current_round += 1
        
        # Apply score decay to all nodes
        for node_id in self.node_scores:
            self.node_scores[node_id] *= self.score_decay_factor
        
        # Update committee terms
        for member_id in self.committee_members:
            self.committee_terms[member_id] += 1
        
        # Reset terms for non-committee members
        for node_id in range(self.total_clients):
            if node_id not in self.committee_members:
                self.committee_terms[node_id] = 0
        
        # Re-elect committee
        self.committee_members = self.elect_committee()

