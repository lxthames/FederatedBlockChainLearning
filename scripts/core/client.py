"""
Federated Client

Represents a client in the federated learning system.
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict


class FederatedClient:
    """
    Represents a client in the federated learning system.
    Each client has its own local model and data.
    """
    
    def __init__(self, client_id: int, model: nn.Module, local_data: Tuple, 
                 device: torch.device, val_split_ratio: float = 0.2):
        """
        Initialize a federated client.
        
        Args:
            client_id: Unique identifier for the client
            model: Local model instance
            local_data: Tuple of (X_train, Y_train)
            device: Device to run training on
            val_split_ratio: Ratio of data to use for validation
        """
        self.client_id = client_id
        self.model = model
        self.device = device
        self.local_epochs = 0
        
        # Split data into train and validation
        X_train, Y_train = local_data
        n_train = int(len(X_train) * (1 - val_split_ratio))
        
        self.train_data = (X_train[:n_train], Y_train[:n_train])
        self.val_data = (X_train[n_train:], Y_train[n_train:])
        self.data_size = len(self.train_data[0])
    
    def train_local(self, epochs: int, batch_size: int = 128, lr: float = 0.001) -> float:
        """
        Train model locally on client data.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
            
        Returns:
            Average training loss
        """
        X_train, Y_train = self.train_data
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
        """
        Get the difference between local and global model.
        Only includes floating-point tensors (excludes integer tensors like num_batches_tracked).
        
        Args:
            global_model_state: State dictionary of the global model
            
        Returns:
            Dictionary of weight differences
        """
        local_state = self.model.state_dict()
        update = {}
        
        for key in local_state:
            # Only include float tensors (skip integer tensors)
            if isinstance(local_state[key], torch.Tensor) and local_state[key].dtype.is_floating_point:
                update[key] = local_state[key] - global_model_state[key]
        
        return update
    
    def apply_global_update(self, global_model_state: Dict):
        """
        Apply global model update to local model.
        
        Args:
            global_model_state: State dictionary of the global model
        """
        self.model.load_state_dict(global_model_state)
    
    def validate_peer_update(self, candidate_weights: Dict, global_weights: Dict) -> Tuple[float, float]:
        """
        Validate a peer's update on local validation data.
        Computes loss and accuracy improvement relative to global model.
        
        Args:
            candidate_weights: Candidate model weights to evaluate
            global_weights: Global model weights for comparison
            
        Returns:
            (u_loss, u_acc) - Utility scores for loss and accuracy
        """
        X_val, Y_val = self.val_data
        val_dataset = TensorDataset(
            torch.from_numpy(X_val).float(),
            torch.from_numpy(Y_val)
        )
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
        
        criterion = nn.MSELoss()
        
        # Evaluate global model
        self.model.load_state_dict(global_weights)
        self.model.eval()
        global_loss = 0.0
        global_correct = 0
        global_total = 0
        
        with torch.no_grad():
            for Xb, Yb in val_loader:
                Xb = Xb.to(self.device)
                Yb = Yb.to(self.device)
                preds = self.model(Xb)
                global_loss += criterion(preds, Yb).item()
                # Accuracy: count correct predictions (argmax)
                global_correct += (preds.argmax(dim=1) == Yb.argmax(dim=1)).sum().item()
                global_total += len(Yb)
        
        global_loss /= len(val_loader)
        global_acc = global_correct / global_total if global_total > 0 else 0.0
        
        # Evaluate candidate model
        self.model.load_state_dict(candidate_weights)
        self.model.eval()
        candidate_loss = 0.0
        candidate_correct = 0
        candidate_total = 0
        
        with torch.no_grad():
            for Xb, Yb in val_loader:
                Xb = Xb.to(self.device)
                Yb = Yb.to(self.device)
                preds = self.model(Xb)
                candidate_loss += criterion(preds, Yb).item()
                candidate_correct += (preds.argmax(dim=1) == Yb.argmax(dim=1)).sum().item()
                candidate_total += len(Yb)
        
        candidate_loss /= len(val_loader)
        candidate_acc = candidate_correct / candidate_total if candidate_total > 0 else 0.0
        
        # Calculate utility scores
        # u_loss: improvement in loss (positive = better)
        u_loss = max(0, global_loss - candidate_loss) / (global_loss + 1e-6)
        
        # u_acc: improvement in accuracy (positive = better)
        u_acc = max(0, candidate_acc - global_acc)
        
        return u_loss, u_acc

