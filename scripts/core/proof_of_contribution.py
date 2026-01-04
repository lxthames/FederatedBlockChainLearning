"""
Proof-of-Contribution (PoC)

Evaluates update utility based on validation accuracy, loss improvement, and robustness.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional


class ProofOfContribution:
    """
    Proof-of-Contribution mechanism to evaluate update utility.
    Metrics: validation accuracy improvement, loss reduction, robustness.
    """
    
    def __init__(self, validation_data: Optional[Tuple] = None):
        """
        Initialize PoC system.
        
        Args:
            validation_data: Optional tuple of (X_val, Y_val) for evaluation
        """
        self.validation_data = validation_data
        self.contribution_history = {}  # {client_id: [contributions]}
        self.baseline_metrics = {}  # Baseline for comparison
    
    def evaluate_contribution(self, client_id: int, model_update: Dict, 
                             local_model: nn.Module, global_model: nn.Module,
                             validation_data: Optional[Tuple] = None,
                             device: torch.device = None) -> Dict[str, float]:
        """
        Evaluate contribution of an update using multiple metrics.
        
        Args:
            client_id: ID of the client
            model_update: Dictionary of model weight updates
            local_model: Local model after training
            global_model: Global model
            validation_data: Optional validation data (X_val, Y_val)
            device: Device to run evaluation on
            
        Returns:
            Dictionary with metrics:
            {
                'accuracy_improvement': float,
                'loss_reduction': float,
                'robustness_score': float,
                'overall_score': float
            }
        """
        if device is None:
            device = torch.device("cpu")
        
        val_data = validation_data or self.validation_data
        if val_data is None:
            # Fallback to simple metrics
            return {
                'accuracy_improvement': 0.0,
                'loss_reduction': 0.0,
                'robustness_score': 0.0,
                'overall_score': 0.0
            }
        
        X_val, Y_val = val_data
        
        # 1. Accuracy Improvement
        global_model.eval()
        local_model.eval()
        
        with torch.no_grad():
            X_val_tensor = torch.from_numpy(X_val).float().to(device)
            Y_val_tensor = torch.from_numpy(Y_val).float().to(device)
            
            # Evaluate global model
            global_preds = global_model(X_val_tensor)
            global_loss = nn.MSELoss()(global_preds, Y_val_tensor).item()
            
            # Evaluate local model (with update applied)
            local_preds = local_model(X_val_tensor)
            local_loss = nn.MSELoss()(local_preds, Y_val_tensor).item()
            
            # Calculate accuracy improvement (using MSE as proxy - lower is better)
            accuracy_improvement = max(0, global_loss - local_loss)
            
            # Loss reduction (normalized)
            loss_reduction = max(0, global_loss - local_loss) / (global_loss + 1e-6)
        
        # 2. Robustness Score (based on update norm and consistency)
        update_norm = sum(
            torch.norm(v).item() 
            for v in model_update.values() 
            if isinstance(v, torch.Tensor) and v.dtype.is_floating_point
        )
        # Smaller, consistent updates are more robust
        robustness_score = 1.0 / (1.0 + update_norm / 100.0)
        
        # 3. Overall Contribution Score
        overall_score = (
            0.4 * accuracy_improvement +
            0.4 * loss_reduction +
            0.2 * robustness_score
        )
        # #region agent log
        import json
        import time
        import os
        try:
            log_path = r'c:\Users\user\Desktop\FederatedBlockchainLearning\.cursor\debug.log'
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({"id": f"log_{int(time.time()*1000)}", "timestamp": int(time.time()*1000), "location": "proof_of_contribution.py:106", "message": "PoC score calculation", "data": {"client_id": client_id, "accuracy_improvement": accuracy_improvement, "loss_reduction": loss_reduction, "robustness_score": robustness_score, "overall_score": overall_score, "global_loss": global_loss, "local_loss": local_loss, "update_norm": update_norm}, "sessionId": "debug-session", "runId": "run1", "hypothesisId": "B"}) + '\n')
        except Exception as e:
            print(f"DEBUG LOG ERROR: {e}")
        # #endregion
        
        # Record contribution
        if client_id not in self.contribution_history:
            self.contribution_history[client_id] = []
        
        self.contribution_history[client_id].append({
            'accuracy_improvement': accuracy_improvement,
            'loss_reduction': loss_reduction,
            'robustness_score': robustness_score,
            'overall_score': overall_score,
            'timestamp': time.time()
        })
        
        return {
            'accuracy_improvement': accuracy_improvement,
            'loss_reduction': loss_reduction,
            'robustness_score': robustness_score,
            'overall_score': overall_score
        }
    
    def rank_updates(self, contributions: Dict[int, Dict[str, float]], 
                    top_k: Optional[int] = None) -> List[Tuple[int, float]]:
        """
        Rank updates by contribution score.
        
        Args:
            contributions: Dictionary of {client_id: contribution_metrics}
            top_k: Optional limit on number of top updates to return
            
        Returns:
            List of (client_id, overall_score) tuples sorted descending
        """
        ranked = sorted(
            [(cid, contrib['overall_score']) for cid, contrib in contributions.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        if top_k:
            return ranked[:top_k]
        return ranked
    
    def meets_threshold(self, contribution_score: float, threshold: float = 0.1) -> bool:
        """
        Check if contribution meets minimum threshold.
        
        Args:
            contribution_score: Overall contribution score
            threshold: Minimum threshold value
            
        Returns:
            True if score meets threshold
        """
        return contribution_score >= threshold
    
    def get_average_contribution(self, client_id: int) -> float:
        """
        Get average contribution score for a client.
        
        Args:
            client_id: ID of the client
            
        Returns:
            Average contribution score
        """
        if client_id not in self.contribution_history:
            return 0.0
        scores = [c['overall_score'] for c in self.contribution_history[client_id]]
        return np.mean(scores) if scores else 0.0

