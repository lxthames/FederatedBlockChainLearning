"""
Script Integration Guide: How to use scripts folder components in the notebook

This shows how to properly integrate all scripts into the experiment.
"""

# ============================================================================
# Step 1: Import Scripts at the Top of Notebook
# ============================================================================
# Add to your imports cell (after standard imports):

import sys
import os

# Add scripts directory to path
scripts_path = os.path.join(os.getcwd(), 'scripts')
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

# Import components as needed
from metrics_logger import MetricsLogger
from second_layer_security import (
    StakingSystem, PoisoningDetector, RollbackBlockchain, SecondLayerSecurity
)

# Optional: Import helper functions
# from poc_debugging import get_adaptive_poc_threshold


# ============================================================================
# Step 2: Load Test Data (SIFT_test_query.npy)
# ============================================================================
# In main_blockchain_federated(), after loading training data:

def load_test_data(base_dir: str, test_file: str = "SIFT_test_query.npy"):
    """
    Load test data for evaluation.
    
    Args:
        base_dir: Base directory containing data files
        test_file: Name of test query file
    
    Returns:
        (X_test, Y_test) or None if file doesn't exist
    """
    test_path = os.path.join(base_dir, test_file)
    
    if not os.path.exists(test_path):
        print(f"⚠️ Test file not found: {test_path}")
        print("   Using validation set for testing instead")
        return None
    
    try:
        X_test = np.load(test_path)
        print(f"✓ Loaded test data: {X_test.shape}")
        
        # If test file doesn't have labels, we'll need to generate them
        # or use a separate label file
        # For now, assume test data is just features
        # You may need to create Y_test separately
        
        return X_test
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return None


# ============================================================================
# Step 3: Setup Test Data in main_blockchain_federated()
# ============================================================================
# Update your main function:

def main_blockchain_federated():
    # ... existing setup ...
    
    # Load test data
    X_test = load_test_data(BASE_DIR, "SIFT_test_query.npy")
    
    # If test data doesn't have labels, create dummy labels for testing
    # Or use a separate label file if available
    if X_test is not None:
        # Option 1: Use dummy labels (for testing only)
        # Y_test = np.random.rand(len(X_test), num_clients).astype(np.float32)
        # Y_test = Y_test / Y_test.sum(axis=1, keepdims=True)
        
        # Option 2: Use validation set labels (if same distribution)
        # Y_test = Y_val  # If test and val have same number of samples
        
        # Option 3: Load from separate file if available
        # Y_test = np.load(os.path.join(BASE_DIR, "SIFT_test_labels.npy"))
        
        test_data = (X_test, Y_test) if 'Y_test' in locals() else None
    else:
        test_data = (X_val, Y_val)  # Fallback to validation set
    
    # Initialize metrics logger with test data
    metrics_logger = MetricsLogger(
        num_clients=num_clients,
        test_data=test_data
    )
    
    # ... rest of setup ...


# ============================================================================
# Step 4: Use Scripts in Training Loop
# ============================================================================
# The scripts are imported as modules, so you can use them directly:

# Example: Using MetricsLogger
metrics_logger = MetricsLogger(num_clients=10, test_data=(X_test, Y_test))

# Example: Using Second Layer Security
from second_layer_security import SecondLayerSecurity
# Initialize in EnhancedBlockchainFederatedLearning.__init__

# Example: Using adaptive threshold function
def get_adaptive_poc_threshold(round_num, base_threshold=0.1, min_threshold=0.01):
    """Copy from poc_debugging.py or import directly"""
    if round_num <= 3:
        return min_threshold
    elif round_num <= 10:
        progress = (round_num - 3) / 7
        return min_threshold + (base_threshold - min_threshold) * progress
    else:
        return base_threshold

