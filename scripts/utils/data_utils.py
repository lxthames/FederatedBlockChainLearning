"""
Data Utilities

Functions for loading, distributing, and evaluating data.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import List, Tuple, Optional

__all__ = [
    'load_test_query_data',
    'load_test_query_with_ground_truth',
    'setup_test_data_for_metrics',
    'distribute_data_to_clients',
    'evaluate_global_model'
]


def load_test_query_data(base_dir: str = "./data", test_file: str = "SIFT_test_query.npy") -> Optional[np.ndarray]:
    """
    Load test query data for evaluation.
    
    Args:
        base_dir: Base directory containing data files
        test_file: Name of test query file
        
    Returns:
        Test query features array, or None if not found
    """
    test_path = os.path.join(base_dir, test_file)
    
    if not os.path.exists(test_path):
        print(f"⚠️ Test query file not found: {test_path}")
        print("   Will use validation set for testing instead")
        return None
    
    try:
        X_test = np.load(test_path)
        print(f"✓ Loaded test query data: {X_test.shape}")
        print(f"   Test samples: {len(X_test)}")
        return X_test
    except Exception as e:
        print(f"❌ Error loading test query: {e}")
        return None


def load_test_query_with_ground_truth(base_dir: str, num_clients: int, 
                                     dist_type: str = "skewed") -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Load test query data with corresponding ground truth labels.
    
    Args:
        base_dir: Base directory containing data files
        num_clients: Number of clients (for determining ground truth file)
        dist_type: Distribution type ("skewed" or "uniform")
        
    Returns:
        (X_test, Y_test) tuple if files exist, None otherwise
    """
    test_query_path = os.path.join(base_dir, "SIFT_test_query.npy")
    ground_truth_path = os.path.join(base_dir, f"gt_{num_clients}M_client_counts_{dist_type}test.npy")
    
    if not os.path.exists(test_query_path):
        print(f"⚠️ Test query file not found: {test_query_path}")
        return None
    
    if not os.path.exists(ground_truth_path):
        print(f"⚠️ Ground truth file not found: {ground_truth_path}")
        return None
    
    try:
        X_test = np.load(test_query_path)
        Y_test_raw = np.load(ground_truth_path)
        
        # Ensure same length
        min_len = min(len(X_test), len(Y_test_raw))
        X_test = X_test[:min_len]
        Y_test_raw = Y_test_raw[:min_len]
        
        # Normalize Y (convert to probabilities)
        Y_sums = Y_test_raw.sum(axis=1, keepdims=True).astype(np.float32)
        Y_sums[Y_sums == 0] = 1.0
        Y_test = Y_test_raw.astype(np.float32) / Y_sums
        
        print(f"✓ Loaded test query data: {X_test.shape}")
        print(f"✓ Loaded test ground truth: {Y_test.shape}")
        return (X_test, Y_test)
    except Exception as e:
        print(f"❌ Error loading test query/ground truth: {e}")
        return None


def setup_test_data_for_metrics(X_test_query: Optional[np.ndarray], 
                                Y_test_query: Optional[np.ndarray],
                                X_val: np.ndarray, 
                                Y_val: np.ndarray, 
                                num_clients: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Setup test data for metrics logger.
    If test query with ground truth is available, use it; otherwise use validation set.
    
    Args:
        X_test_query: Test query features (from SIFT_test_query.npy) or None
        Y_test_query: Test query ground truth labels or None
        X_val: Validation features
        Y_val: Validation labels
        num_clients: Number of clients (for label shape)
        
    Returns:
        (X_test, Y_test) tuple for metrics logger
    """
    if X_test_query is not None and Y_test_query is not None:
        print(f"✓ Using test query data with ground truth: {X_test_query.shape}")
        return (X_test_query, Y_test_query)
    else:
        print(f"Using validation set as test data: {X_val.shape}")
        return (X_val, Y_val)


def distribute_data_to_clients(X_train: np.ndarray, Y_train: np.ndarray, 
                                num_clients: int, distribution: str = "uniform") -> List[Tuple]:
    """
    Distribute training data across clients.
    
    Args:
        X_train: Training features
        Y_train: Training labels
        num_clients: Number of clients
        distribution: Distribution type ("uniform" or "skewed")
        
    Returns:
        List of (X_client, Y_client) tuples
    """
    np.random.seed(42)
    n_samples = len(X_train)
    client_data_list = []
    
    if distribution == "uniform":
        # Uniform distribution: each client gets roughly equal data
        samples_per_client = n_samples // num_clients
        for i in range(num_clients):
            start_idx = i * samples_per_client
            end_idx = (i + 1) * samples_per_client if i < num_clients - 1 else n_samples
            client_data_list.append((
                X_train[start_idx:end_idx],
                Y_train[start_idx:end_idx]
            ))
    else:  # skewed distribution
        # Skewed: some clients have more data than others
        indices = np.random.permutation(n_samples)
        client_sizes = []
        total = 0
        
        for i in range(num_clients):
            # Exponential decay in data size
            size = int(n_samples * (0.5 ** i) / sum(0.5 ** j for j in range(num_clients)))
            client_sizes.append(size)
            total += size
        
        # Adjust last client to use all remaining data
        client_sizes[-1] += n_samples - total
        
        start_idx = 0
        for size in client_sizes:
            end_idx = start_idx + size
            client_indices = indices[start_idx:end_idx]
            client_data_list.append((
                X_train[client_indices],
                Y_train[client_indices]
            ))
            start_idx = end_idx
    
    return client_data_list


def evaluate_global_model(model: nn.Module, X_val: np.ndarray, Y_val: np.ndarray, 
                         device: torch.device, batch_size: int = 256) -> float:
    """
    Evaluate the global model on validation data.
    
    Args:
        model: Model to evaluate
        X_val: Validation features
        Y_val: Validation labels
        device: Device to run evaluation on
        batch_size: Batch size for evaluation
        
    Returns:
        Average validation loss
    """
    val_dataset = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(Y_val)
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    criterion = nn.MSELoss()
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for Xb, Yb in val_loader:
            Xb = Xb.to(device)
            Yb = Yb.to(device)
            preds = model(Xb)
            val_loss += criterion(preds, Yb).item()
    
    return val_loss / len(val_loader)

