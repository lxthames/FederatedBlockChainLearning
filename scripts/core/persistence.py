"""
Chain Persistence

Handles saving and loading blockchain state to/from disk.
"""

import os
import json
from typing import List, Dict, Optional


class ChainPersistence:
    """Persists blockchain to disk for basic durability and cross-process sharing"""
    
    def __init__(self, path: str = "blockchain_state.json"):
        """
        Initialize persistence manager.
        
        Args:
            path: Path to the JSON file for storing blockchain state
        """
        self.path = path
    
    def save(self, chain: List[Dict]):
        """
        Save blockchain state to disk.
        
        Args:
            chain: List of block dictionaries to save
        """
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(chain, f, indent=2)
        except Exception as e:
            print(f"⚠️ Failed to persist chain: {e}")
    
    def load(self) -> Optional[List[Dict]]:
        """
        Load blockchain state from disk.
        
        Returns:
            List of block dictionaries if file exists, None otherwise
        """
        if not os.path.exists(self.path):
            return None
        
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Failed to load persisted chain: {e}")
            return None

