"""
Blockchain Implementation

Contains Block, Blockchain, and SignedBlockchain classes for storing
federated learning model updates in an immutable ledger.
"""

import time
import json
import hashlib
import numpy as np
from typing import List, Dict, Optional
from .persistence import ChainPersistence


class Block:
    """Represents a block in the blockchain storing model updates"""
    
    def __init__(self, index: int, transactions: List[Dict], previous_hash: str, 
                 committee_votes: Dict[int, bool] = None, timestamp: float = None, 
                 stored_hash: str = None):
        """
        Initialize a block.
        
        Args:
            index: Block index in the chain
            transactions: List of model update transactions
            previous_hash: Hash of the previous block
            committee_votes: Dictionary of committee member votes
            timestamp: Optional timestamp (for loading from persistence)
            stored_hash: Optional stored hash (for loading from persistence, preserves original hash)
        """
        self.index = index
        self.transactions = transactions
        self.previous_hash = previous_hash
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.committee_votes = committee_votes or {}
        # Use stored hash if provided (from persistence), otherwise calculate
        self.hash = stored_hash if stored_hash is not None else self.calculate_hash()
    
    def calculate_hash(self) -> str:
        """Calculate SHA256 hash of the block"""
        # Use a more robust serialization that handles numpy types and complex nested structures
        def json_serializer(obj):
            """Custom serializer for complex types"""
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                # Recursively serialize dict values
                return {k: json_serializer(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [json_serializer(item) for item in obj]
            return str(obj)
        
        try:
            block_string = json.dumps({
                "index": self.index,
                "transactions": json_serializer(self.transactions),
                "previous_hash": self.previous_hash,
                "timestamp": self.timestamp,
                "committee_votes": self.committee_votes
            }, sort_keys=True, default=json_serializer)
            return hashlib.sha256(block_string.encode()).hexdigest()
        except (TypeError, ValueError) as e:
            # Fallback: use string representation if JSON serialization fails
            block_string = f"{self.index}{str(self.transactions)}{self.previous_hash}{self.timestamp}{str(self.committee_votes)}"
            return hashlib.sha256(block_string.encode()).hexdigest()
    
    def to_dict(self) -> Dict:
        """Convert block to dictionary for serialization"""
        return {
            "index": self.index,
            "transactions": self.transactions,
            "previous_hash": self.previous_hash,
            "timestamp": self.timestamp,
            "hash": self.hash,
            "committee_votes": self.committee_votes
        }


class Blockchain:
    """Blockchain for storing federated learning model updates"""
    
    def __init__(self):
        """Initialize blockchain with genesis block"""
        self.chain = [self.create_genesis_block()]
        self.pending_transactions = []
        self.committee_members = []
    
    def create_genesis_block(self) -> Block:
        """Create the first block in the chain"""
        return Block(0, [], "0", {})
    
    def get_latest_block(self) -> Block:
        """Get the most recent block"""
        return self.chain[-1]
    
    def add_transaction(self, client_id: int, model_update: Dict, local_epochs: int, 
                       data_size: int, loss: float):
        """
        Add a model update transaction to pending transactions.
        
        Args:
            client_id: ID of the client submitting the update
            model_update: Serialized model weights
            local_epochs: Number of local training epochs
            data_size: Size of client's dataset
            loss: Training loss
        """
        transaction = {
            "client_id": client_id,
            "model_update": model_update,
            "local_epochs": local_epochs,
            "data_size": data_size,
            "loss": loss,
            "timestamp": time.time()
        }
        self.pending_transactions.append(transaction)
    
    def mine_block(self, committee_votes: Dict[int, bool]) -> Block:
        """
        Mine a new block with pending transactions.
        
        Args:
            committee_votes: Dictionary of committee member votes
            
        Returns:
            The newly created block
        """
        block = Block(
            len(self.chain),
            self.pending_transactions.copy(),
            self.get_latest_block().hash,
            committee_votes
        )
        self.chain.append(block)
        self.pending_transactions = []
        return block
    
    def validate_chain(self) -> bool:
        """
        Validate the integrity of the blockchain.
        
        Returns:
            True if chain is valid, False otherwise
        """
        if len(self.chain) <= 1:
            return True  # Only genesis block is always valid
        
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i-1]
            
            # Check hash integrity
            recalculated_hash = current.calculate_hash()
            if current.hash != recalculated_hash:
                # Debug: Print which block failed
                print(f"  ⚠️ Block {current.index} hash mismatch: stored={current.hash[:16]}..., calculated={recalculated_hash[:16]}...")
                return False
            # Check chain linkage
            if current.previous_hash != previous.hash:
                print(f"  ⚠️ Block {current.index} chain linkage broken: prev_hash={current.previous_hash[:16]}..., expected={previous.hash[:16]}...")
                return False
        return True
    
    def get_consensus_threshold(self) -> float:
        """Get the consensus threshold (2/3 majority)"""
        return 2.0 / 3.0
    
    def validate_block_by_committee(self, block: Block) -> bool:
        """
        Validate block through committee consensus.
        
        Args:
            block: Block to validate
            
        Returns:
            True if block passes committee validation
        """
        if not self.committee_members:
            return True  # No committee, accept all
        
        votes = block.committee_votes
        positive_votes = sum(1 for vote in votes.values() if vote)
        threshold = len(self.committee_members) * self.get_consensus_threshold()
        
        return positive_votes >= threshold


class SignedBlockchain(Blockchain):
    """Blockchain that stores signatures and persists state"""
    
    def __init__(self, persistence: ChainPersistence = None):
        """
        Initialize signed blockchain with optional persistence.
        
        Args:
            persistence: ChainPersistence instance for saving/loading
        """
        # Don't create genesis block yet - load from persistence first if available
        self.persistence = persistence
        self.pending_transactions = []
        self.committee_members = []
        
        loaded_chain = None
        if self.persistence:
            loaded_chain = self.persistence.load()
        
        if loaded_chain and len(loaded_chain) > 0:
            # Rebuild chain objects from dicts, preserving original hashes
            self.chain = [
                Block(
                    b["index"], 
                    b["transactions"], 
                    b["previous_hash"], 
                    b.get("committee_votes", {}),
                    timestamp=b.get("timestamp"),  # Preserve original timestamp
                    stored_hash=b.get("hash")  # Preserve original hash to maintain chain integrity
                ) 
                for b in loaded_chain
            ]
            # Validate loaded chain; if invalid, reset to genesis
            if not self.validate_chain():
                print("⚠️ Persisted chain is invalid; resetting to new genesis block.")
                self.chain = [self.create_genesis_block()]
        else:
            # No persisted chain, create genesis block
            self.chain = [self.create_genesis_block()]
    
    def add_transaction(self, client_id: int, model_update: Dict, local_epochs: int, 
                       data_size: int, loss: float, signature: str, public_id: str):
        """
        Add a signed transaction to pending transactions.
        
        Args:
            client_id: ID of the client
            model_update: Serialized model weights
            local_epochs: Number of local training epochs
            data_size: Size of client's dataset
            loss: Training loss
            signature: HMAC signature of the transaction
            public_id: Public identifier of the client
        """
        transaction = {
            "client_id": client_id,
            "public_id": public_id,
            "model_update": model_update,
            "local_epochs": local_epochs,
            "data_size": data_size,
            "loss": loss,
            "timestamp": time.time(),
            "signature": signature,
        }
        self.pending_transactions.append(transaction)
    
    def mine_block(self, committee_votes: Dict[int, bool]) -> Block:
        """
        Mine a new block and persist the chain.
        
        Args:
            committee_votes: Dictionary of committee member votes
            
        Returns:
            The newly created block
        """
        block = super().mine_block(committee_votes)
        
        # Persist chain
        if self.persistence:
            self.persistence.save([b.to_dict() for b in self.chain])
        
        return block

