"""
Create matching Y data file for SIFT1B_train.npy

This script creates a Y (target) file with the same number of samples as X.
It can create uniform or skewed distributions.
"""

import numpy as np
import os
import sys

def create_matching_y_data(x_file: str, output_file: str, num_clients: int = 10, 
                          distribution: str = "skewed", seed: int = 42):
    """
    Create Y data file matching the size of X data file.
    
    Args:
        x_file: Path to SIFT1B_train.npy (X features)
        output_file: Path to save the Y data file
        num_clients: Number of clients (output dimension)
        distribution: "uniform" or "skewed"
        seed: Random seed for reproducibility
    """
    print(f"Loading X data from: {x_file}")
    X = np.load(x_file)
    num_samples = len(X)
    print(f"X shape: {X.shape}")
    print(f"Creating Y data with {num_samples} samples for {num_clients} clients...")
    
    np.random.seed(seed)
    
    if distribution == "uniform":
        # Uniform distribution: each client gets roughly equal probability
        Y = np.random.rand(num_samples, num_clients).astype(np.float32)
        # Normalize to sum to 1 (probability distribution)
        Y = Y / Y.sum(axis=1, keepdims=True)
        print("Created uniform distribution")
    else:  # skewed
        # Skewed distribution: exponential decay
        Y = np.random.exponential(scale=2.0, size=(num_samples, num_clients)).astype(np.float32)
        # Normalize to sum to 1
        Y = Y / Y.sum(axis=1, keepdims=True)
        print("Created skewed distribution")
    
    # Save the file
    print(f"Saving Y data to: {output_file}")
    np.save(output_file, Y)
    print(f"✓ Saved Y data: shape {Y.shape}")
    print(f"  File size: {os.path.getsize(output_file) / (1024**2):.2f} MB")
    
    return Y


if __name__ == "__main__":
    # Default paths
    BASE_DIR = "./data"
    X_FILE = os.path.join(BASE_DIR, "SIFT1B_train.npy")
    
    # Configuration
    NUM_CLIENTS = 10
    DISTRIBUTION = "skewed"  # or "uniform"
    
    # Output file name
    OUTPUT_FILE = os.path.join(BASE_DIR, f"gt_{NUM_CLIENTS}M_client_counts_{DISTRIBUTION}_full.npy")
    
    # Check if X file exists
    if not os.path.exists(X_FILE):
        print(f"❌ Error: X file not found: {X_FILE}")
        print(f"   Please update X_FILE path in the script")
        sys.exit(1)
    
    # Create matching Y data
    Y = create_matching_y_data(
        x_file=X_FILE,
        output_file=OUTPUT_FILE,
        num_clients=NUM_CLIENTS,
        distribution=DISTRIBUTION,
        seed=42
    )
    
    print("\n" + "="*60)
    print("SUCCESS!")
    print("="*60)
    print(f"\nCreated: {OUTPUT_FILE}")
    print(f"Shape: {Y.shape}")
    print(f"\nTo use this file, update your notebook:")
    print(f'  "train_target_y_npy": rf"{OUTPUT_FILE}"')
    print("\nOr rename it to match your current naming:")
    print(f'  Rename: {OUTPUT_FILE}')
    print(f'  To:     {os.path.join(BASE_DIR, f"gt_{NUM_CLIENTS}M_client_counts_{DISTRIBUTION}.npy")}')

