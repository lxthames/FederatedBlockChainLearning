"""
Proof-of-Personhood (PoP)

Prevents Sybil attacks by ensuring one identity per participant.
Uses web-of-trust and challenge verification.
"""

import time
import secrets
from typing import Tuple, Dict
from collections import defaultdict


class ProofOfPersonhood:
    """
    Proof-of-Personhood mechanism to prevent Sybil attacks.
    Simplified implementation using web-of-trust and challenge verification.
    """
    
    def __init__(self, min_trust_score: float = 0.5):
        """
        Initialize PoP system.
        
        Args:
            min_trust_score: Minimum trust score required to participate
        """
        self.min_trust_score = min_trust_score
        self.registered_nodes = {}  # {node_id: registration_info}
        self.trust_graph = defaultdict(set)  # {node_id: {trusted_nodes}}
        self.challenge_history = {}  # {node_id: [challenges]}
        self.verification_tokens = {}  # {node_id: token}
    
    def register_node(self, node_id: int, verification_method: str = "challenge") -> Tuple[bool, str]:
        """
        Register a new node with PoP verification.
        
        Args:
            node_id: ID of the node to register
            verification_method: Method of verification ("challenge" or "web_of_trust")
            
        Returns:
            (success, token_or_error_message)
        """
        if node_id in self.registered_nodes:
            return False, "Node already registered"
        
        if verification_method == "challenge":
            # Generate a challenge token (simplified - in practice, use CAPTCHA or ZK proof)
            token = secrets.token_hex(16)
            self.verification_tokens[node_id] = token
            self.challenge_history[node_id] = []
            
            # Simulate challenge completion (in practice, node solves CAPTCHA/ZK proof)
            challenge_completed = True  # Simplified
            
            if challenge_completed:
                self.registered_nodes[node_id] = {
                    "registration_time": time.time(),
                    "trust_score": 0.5,  # Bootstrapping: Initial trust allows first participation (probation period)
                    "verification_method": verification_method,
                    "verified": True,
                    "probation": True  # Mark as in probation period
                }
                return True, token
            else:
                return False, "Challenge verification failed"
        
        elif verification_method == "web_of_trust":
            # Web-of-trust: requires endorsement from existing trusted nodes
            # Simplified: auto-register first N nodes, then require endorsements
            if len(self.registered_nodes) < 3:
                self.registered_nodes[node_id] = {
                    "registration_time": time.time(),
                    "trust_score": 1.0,
                    "verification_method": verification_method,
                    "verified": True,
                    "endorsers": []
                }
                return True, "auto_registered"
            else:
                # Require at least 2 endorsements from registered nodes
                return False, "Requires web-of-trust endorsements"
        
        return False, "Unknown verification method"
    
    def verify_node(self, node_id: int) -> bool:
        """
        Verify if a node is registered and verified.
        
        Args:
            node_id: ID of the node to verify
            
        Returns:
            True if node is registered and verified
        """
        return (node_id in self.registered_nodes and 
                self.registered_nodes[node_id].get("verified", False))
    
    def endorse_node(self, endorser_id: int, node_id: int) -> bool:
        """
        Endorse a node in web-of-trust (for future registration).
        
        Args:
            endorser_id: ID of the endorsing node
            node_id: ID of the node being endorsed
            
        Returns:
            True if endorsement successful
        """
        if not self.verify_node(endorser_id):
            return False
        
        if node_id not in self.trust_graph:
            self.trust_graph[node_id] = set()
        
        self.trust_graph[node_id].add(endorser_id)
        
        # Update trust score based on endorsements
        if node_id in self.registered_nodes:
            endorsements = len(self.trust_graph[node_id])
            self.registered_nodes[node_id]["trust_score"] = min(1.0, 0.5 + 0.1 * endorsements)
        
        return True
    
    def get_trust_score(self, node_id: int) -> float:
        """
        Get trust score of a node.
        
        Args:
            node_id: ID of the node
            
        Returns:
            Trust score (0.0 to 1.0)
        """
        if node_id not in self.registered_nodes:
            return 0.0
        return self.registered_nodes[node_id].get("trust_score", 0.0)
    
    def can_participate(self, node_id: int) -> bool:
        """
        Check if node can participate (registered + trust score threshold).
        
        Args:
            node_id: ID of the node
            
        Returns:
            True if node can participate
        """
        if not self.verify_node(node_id):
            return False
        return self.get_trust_score(node_id) >= self.min_trust_score
    
    def update_trust_score(self, node_id: int, new_score: float):
        """
        Update trust score of a node (e.g., after good/bad behavior).
        
        Args:
            node_id: ID of the node
            new_score: New trust score (clamped to [0, 1])
        """
        if node_id in self.registered_nodes:
            self.registered_nodes[node_id]["trust_score"] = max(0.0, min(1.0, new_score))

