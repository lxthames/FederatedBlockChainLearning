"""
Refactored Components for Client-Local Validation & Committee Median Voting

Copy these implementations into your notebook to replace the existing classes.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Dict, Tuple, Optional
import copy

# ============================================================================
# PoC Configuration Constants (from paper)
# ============================================================================
GAMMA_1 = 0.4  # Weight for median(u_loss) in PoC score
GAMMA_2 = 0.4  # Weight for median(u_acc) in PoC score
GAMMA_3 = 0.2  # Weight for u_rob (robustness) in PoC score
DELTA_TH = 0.1  # Threshold for update acceptance
LAMBDA = 0.01  # Lambda parameter for robustness calculation
VAL_SPLIT_RATIO = 0.2  # 20% of local data for validation


# ============================================================================
# Modified FederatedClient Class
# ============================================================================
class FederatedClient:
    """Represents a client in the federated learning system with local validation"""
    def __init__(self, client_id: int, model: nn.Module, local_data: Tuple, 
                 device: torch.device, val_split_ratio: float = VAL_SPLIT_RATIO):
        self.client_id = client_id
        self.model = model
        self.device = device
        self.local_epochs = 0
        
        # Split local data into train and validation sets
        X_full, Y_full = local_data
        n_samples = len(X_full)
        n_val = int(n_samples * val_split_ratio)
        
        # Random split
        np.random.seed(42 + client_id)  # Different seed per client for variety
        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        self.train_data = (X_full[train_indices], Y_full[train_indices])
        self.val_data = (X_full[val_indices], Y_full[val_indices])
        self.data_size = len(self.train_data[0])
        
        print(f"Client {client_id}: Train={len(self.train_data[0])}, Val={len(self.val_data[0])}")
    
    def validate_peer_update(self, candidate_weights: Dict, global_weights: Dict, 
                            device: torch.device = None) -> Tuple[float, float]:
        """
        Validate a peer's update on this client's local validation set.
        
        Computes Loss L(w_k) and Accuracy Acc(w_k) on self.val_data.
        Calculates u_loss and u_acc relative to the global model's performance.
        
        Args:
            candidate_weights: Model weights with the update applied
            global_weights: Current global model weights
            device: Device to use for evaluation
            
        Returns:
            (u_loss, u_acc): Utility scores for loss and accuracy
        """
        if device is None:
            device = self.device
        
        X_val, Y_val = self.val_data
        criterion = nn.MSELoss()
        
        # Create temporary models for evaluation
        candidate_model = copy.deepcopy(self.model)
        global_model = copy.deepcopy(self.model)
        
        candidate_model.load_state_dict(candidate_weights)
        global_model.load_state_dict(global_weights)
        
        candidate_model.eval()
        global_model.eval()
        
        with torch.no_grad():
            X_val_tensor = torch.from_numpy(X_val).float().to(device)
            Y_val_tensor = torch.from_numpy(Y_val).float().to(device)
            
            # Evaluate candidate model
            candidate_preds = candidate_model(X_val_tensor)
            candidate_loss = criterion(candidate_preds, Y_val_tensor).item()
            
            # Calculate accuracy (1 - normalized MSE, higher is better)
            # For multi-output regression, use inverse of normalized MSE
            mse = candidate_loss
            max_possible_mse = torch.max(Y_val_tensor).item() ** 2 if len(Y_val_tensor) > 0 else 1.0
            candidate_acc = max(0, 1.0 - mse / (max_possible_mse + 1e-6))
            
            # Evaluate global model
            global_preds = global_model(X_val_tensor)
            global_loss = criterion(global_preds, Y_val_tensor).item()
            global_mse = global_loss
            global_acc = max(0, 1.0 - global_mse / (max_possible_mse + 1e-6))
            
            # Calculate utility scores
            # u_loss: improvement in loss (higher is better)
            # u_loss = max(0, global_loss - candidate_loss) / (global_loss + 1e-6)
            u_loss = max(0, global_loss - candidate_loss) / (global_loss + 1e-6)
            
            # u_acc: improvement in accuracy (higher is better)
            u_acc = max(0, candidate_acc - global_acc)
        
        return (u_loss, u_acc)
    
    def train_local(self, epochs: int, batch_size: int = 128, lr: float = 0.001):
        """Train model locally on client's training data"""
        X_train, Y_train = self.train_data  # Use train_data instead of local_data
        train_dataset = TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(Y_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        
        self.model.train()
        total_loss = 0.0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for Xb, Yb in train_loader:
                Xb = Xb.to(self.device)
                Yb = Yb.to(self.device)
                
                optimizer.zero_grad()
                preds = self.model(Xb)
                loss = criterion(preds, Yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
            
            total_loss += epoch_loss / len(train_loader)
        
        self.local_epochs += epochs
        avg_loss = total_loss / epochs
        return avg_loss
    
    def get_model_update(self, global_model_state: Dict) -> Dict:
        """Get the difference between local and global model"""
        local_state = self.model.state_dict()
        update = {}
        
        for key in local_state:
            # Only include float tensors (skip integer tensors like num_batches_tracked)
            if isinstance(local_state[key], torch.Tensor) and local_state[key].dtype.is_floating_point:
                update[key] = local_state[key] - global_model_state[key]
        
        return update
    
    def apply_global_update(self, global_model_state: Dict):
        """Apply global model update to local model"""
        self.model.load_state_dict(global_model_state)


# ============================================================================
# Simplified ProofOfContribution Class (No Global Validation)
# ============================================================================
class ProofOfContribution:
    """
    Proof-of-Contribution mechanism - now uses committee voting.
    No global validation set needed.
    """
    def __init__(self):
        self.contribution_history = {}  # {client_id: [contributions]}
    
    def calculate_robustness(self, model_update: Dict, lambda_param: float = LAMBDA) -> float:
        """
        Calculate robustness score based on update norm.
        This is a global calculation, not committee-based.
        
        Formula: u_rob = 1 / (1 + lambda * ||Delta w_k||)
        """
        update_norm = sum(
            torch.norm(v).item() 
            for v in model_update.values() 
            if isinstance(v, torch.Tensor) and v.dtype.is_floating_point
        )
        # Robustness: smaller updates are more robust
        u_rob = 1.0 / (1.0 + lambda_param * update_norm)
        return u_rob
    
    def meets_threshold(self, contribution_score: float, threshold: float = DELTA_TH) -> bool:
        """Check if contribution meets minimum threshold"""
        return contribution_score >= threshold
    
    def get_average_contribution(self, client_id: int) -> float:
        """Get average contribution score for a client"""
        if client_id not in self.contribution_history:
            return 0.0
        scores = [c.get('poc_score', 0.0) for c in self.contribution_history[client_id]]
        return np.mean(scores) if scores else 0.0


# ============================================================================
# Helper Function for Committee-Based PoC Scoring
# ============================================================================
def calculate_poc_score_with_committee(
    median_u_loss: float,
    median_u_acc: float,
    u_rob: float,
    gamma_1: float = GAMMA_1,
    gamma_2: float = GAMMA_2,
    gamma_3: float = GAMMA_3
) -> float:
    """
    Calculate PoC score using median of committee votes.
    
    Formula: S_PoC(k) = gamma_1 * Median(u_loss) + gamma_2 * Median(u_acc) + gamma_3 * u_rob
    """
    return gamma_1 * median_u_loss + gamma_2 * median_u_acc + gamma_3 * u_rob

