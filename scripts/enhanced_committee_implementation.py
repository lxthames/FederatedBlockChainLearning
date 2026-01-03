"""
Enhanced Committee Consensus Implementation
Copy this code to replace the EnhancedCommitteeConsensus class in your notebook.
"""

from typing import List, Dict, Tuple, Optional

class EnhancedCommitteeConsensus(CommitteeConsensus):
    """Enhanced committee consensus with scoring, decay, and term limits"""
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
        self.election_method = election_method  # "random", "score", "multi_factor"
        self.node_scores = {i: 0.0 for i in range(total_clients)}  # Historical scores
        self.validation_scores = {}  # {round: {client_id: score}}
        self.current_round = 0
        self.score_decay_factor = score_decay_factor  # Decay factor for historical scores
        self.term_limit = term_limit  # Term limit for committee membership
        self.committee_history = {}  # {client_id: [round_numbers]} - tracks when nodes were in committee
        self.consecutive_terms = {i: 0 for i in range(total_clients)}  # Consecutive rounds in committee
    
    def validate_update_with_score(self, model_update: Dict, validation_metrics: Dict, 
                                   validation_data: Tuple = None) -> Tuple[bool, float]:
        """
        Validate update and return score
        Returns: (accepted, score)
        """
        loss = validation_metrics.get('loss', float('inf'))
        update_norm = validation_metrics.get('update_norm', float('inf'))
        
        # Calculate validation score (higher is better)
        # Score based on loss (lower loss = higher score) and update norm
        if loss < 1.0 and update_norm < 100.0:
            # Score: inverse of loss, penalized by update norm
            score = max(0, (1.0 / (loss + 1e-6)) * (1.0 / (update_norm + 1.0)))
            accepted = True
        else:
            score = 0.0
            accepted = False
        
        return accepted, score
    
    def record_validation_score(self, client_id: int, score: float):
        """Record validation score for a client"""
        if self.current_round not in self.validation_scores:
            self.validation_scores[self.current_round] = {}
        self.validation_scores[self.current_round][client_id] = score
        # Update historical score (moving average)
        self.node_scores[client_id] = 0.7 * self.node_scores[client_id] + 0.3 * score
    
    def elect_committee(self, method: str = None) -> List[int]:
        """Elect committee using specified method with term limits"""
        method = method or self.election_method
        
        if method == "random":
            committee = self.select_committee()  # Use parent's random selection
        elif method == "score":
            # Select top performers based on scores
            sorted_nodes = sorted(
                self.node_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            committee = [node_id for node_id, _ in sorted_nodes[:self.committee_size]]
        elif method == "multi_factor":
            # Multi-factor optimization (simplified: score + participation)
            # In practice, could include network bandwidth, computing capacity, etc.
            participation = {i: len([r for r in self.validation_scores.values() 
                                    if i in r]) for i in range(self.total_clients)}
            
            # Combined score: validation score + participation bonus
            combined_scores = {
                i: self.node_scores[i] + 0.1 * participation.get(i, 0)
                for i in range(self.total_clients)
            }
            sorted_nodes = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            committee = [node_id for node_id, _ in sorted_nodes[:self.committee_size]]
        else:
            committee = self.select_committee()
        
        # Apply term limits if enabled
        if self.term_limit is not None:
            committee = self._apply_term_limits(committee)
        
        return sorted(committee)
    
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
            # Get all nodes sorted by score, excluding already selected and excluded
            all_nodes = sorted(
                [(i, self.node_scores[i]) for i in range(self.total_clients) 
                 if i not in final_committee and i not in excluded],
                key=lambda x: x[1],
                reverse=True
            )
            
            # Add nodes until committee is full
            for node_id, _ in all_nodes:
                if len(final_committee) >= self.committee_size:
                    break
                if self.consecutive_terms[node_id] < self.term_limit:
                    final_committee.append(node_id)
        
        return sorted(final_committee)
    
    def start_new_round(self):
        """Start a new training round, apply score decay, and re-elect committee"""
        self.current_round += 1
        
        # Apply decay factor to historical scores (so old scores fade over time)
        # This gives new high-performing nodes a chance to enter the committee
        for client_id in self.node_scores:
            self.node_scores[client_id] *= self.score_decay_factor
        
        # Update consecutive terms: increment for current committee, reset for others
        for client_id in range(self.total_clients):
            if client_id in self.committee_members:
                self.consecutive_terms[client_id] += 1
                # Track committee history
                if client_id not in self.committee_history:
                    self.committee_history[client_id] = []
                self.committee_history[client_id].append(self.current_round - 1)
            else:
                # Reset consecutive terms for nodes not in committee
                self.consecutive_terms[client_id] = 0
        
        # Re-elect committee
        self.committee_members = self.elect_committee()

