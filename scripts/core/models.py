"""
Neural Network Models

Contains the EnhancedMultiHeadDNN model architecture.
"""

import torch
import torch.nn as nn


class EnhancedMultiHeadDNN(nn.Module):
    """
    Parallel shared representation with multi-head outputs.
    Depth ≤ 3, parallelism without degradation.
    
    Architecture:
    - Two parallel branches for feature extraction
    - Per-client heads producing normalized probabilities
    - Batch normalization and dropout for regularization
    """
    
    def __init__(self, input_size: int, num_clients: int, dropout_rate: float = 0.3):
        """
        Initialize the model.
        
        Args:
            input_size: Size of input features
            num_clients: Number of clients (output heads)
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        self.num_clients = num_clients

        # Parallel shared branches
        self.branch1 = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.branch2 = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
        )

        # Per-client heads
        self.heads = nn.ModuleList([
            nn.Linear(128, 1)
            for _ in range(num_clients)
        ])

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Output tensor of shape (batch_size, num_clients) with normalized probabilities
        """
        # Parallel feature extraction
        f1 = self.branch1(x)
        f2 = self.branch2(x)

        # Merge (summation is more stable than concat)
        features = 0.5 * (f1 + f2)

        # Generate logits for each client
        logits = torch.cat(
            [head(features) for head in self.heads],
            dim=1
        )
        
        # Return normalized probabilities
        return torch.softmax(logits, dim=1)

