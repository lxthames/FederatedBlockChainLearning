"""
Identity Management

Handles client identities and cryptographic signatures.
"""

import hmac
import secrets
import json
import hashlib
from typing import Dict


class IdentityManager:
    """
    Manages per-client identities and signing using HMAC (simulated).
    In production, use proper cryptographic signatures (ECDSA, RSA, etc.)
    """
    
    def __init__(self, num_clients: int):
        """
        Initialize identity manager.
        
        Args:
            num_clients: Number of clients in the system
        """
        self.secrets = {i: secrets.token_bytes(32) for i in range(num_clients)}
        self.public_ids = {i: f"client-{i}" for i in range(num_clients)}
    
    def sign(self, client_id: int, payload: Dict) -> str:
        """
        Sign a payload using client's secret key.
        
        Args:
            client_id: ID of the client
            payload: Dictionary to sign
            
        Returns:
            HMAC signature as hex string
        """
        key = self.secrets[client_id]
        body = json.dumps(payload, sort_keys=True, default=str).encode()
        return hmac.new(key, body, hashlib.sha256).hexdigest()
    
    def verify(self, client_id: int, payload: Dict, signature: str) -> bool:
        """
        Verify a signature.
        
        Args:
            client_id: ID of the client
            payload: Dictionary that was signed
            signature: Signature to verify
            
        Returns:
            True if signature is valid, False otherwise
        """
        expected = self.sign(client_id, payload)
        return hmac.compare_digest(expected, signature)

