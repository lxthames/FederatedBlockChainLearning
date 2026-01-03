"""
Generate sample data for testing the federated learning framework.

Usage:
    python generate_sample_data.py --num_samples 10000 --num_clients 5 --output_dir ./data
"""

import argparse
import numpy as np
import os
from pathlib import Path


def generate_sample_data(num_samples: int, num_clients: int, feature_dim: int = 128, 
                         output_dir: str = "./data", seed: int = 42):
    """
    Generate sample SIFT-like features and client distribution data.
    
    Args:
        num_samples: Number of data samples
        num_clients: Number of clients
        feature_dim: Feature dimension (default: 128 for SIFT)
        output_dir: Output directory for data files
        seed: Random seed for reproducibility
    """
    # Set random seed
    np.random.seed(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating sample data...")
    print(f"  Samples: {num_samples}")
    print(f"  Clients: {num_clients}")
    print(f"  Feature dimension: {feature_dim}")
    
    # Generate SIFT-like features (normal distribution)
    print("\nGenerating features (X)...")
    X = np.random.randn(num_samples, feature_dim).astype(np.float32)
    
    # Normalize features (optional, but common for SIFT)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    
    # Generate client distributions (Y)
    print("Generating client distributions (Y)...")
    Y = np.random.rand(num_samples, num_clients).astype(np.float32)
    
    # Normalize to sum to 1 (probability distribution)
    Y_sums = Y.sum(axis=1, keepdims=True)
    Y_sums[Y_sums == 0] = 1.0  # Avoid division by zero
    Y = Y / Y_sums
    
    # Save feature file
    feature_file = os.path.join(output_dir, "SIFT1B_train.npy")
    np.save(feature_file, X)
    print(f"\n✓ Saved features to: {feature_file}")
    print(f"  Shape: {X.shape}")
    print(f"  Size: {X.nbytes / (1024**2):.2f} MB")
    
    # Save target file (uniform distribution)
    target_file_uniform = os.path.join(output_dir, f"gt_{num_clients}M_client_counts_uniform.npy")
    np.save(target_file_uniform, Y)
    print(f"\n✓ Saved targets (uniform) to: {target_file_uniform}")
    print(f"  Shape: {Y.shape}")
    print(f"  Size: {Y.nbytes / (1024**2):.2f} MB")
    
    # Generate skewed distribution
    print("\nGenerating skewed distribution...")
    Y_skewed = np.random.exponential(scale=2.0, size=(num_samples, num_clients)).astype(np.float32)
    Y_skewed = Y_skewed / Y_skewed.sum(axis=1, keepdims=True)
    
    target_file_skewed = os.path.join(output_dir, f"gt_{num_clients}M_client_counts_skewed.npy")
    np.save(target_file_skewed, Y_skewed)
    print(f"✓ Saved targets (skewed) to: {target_file_skewed}")
    print(f"  Shape: {Y_skewed.shape}")
    
    print("\n" + "="*60)
    print("Sample data generation complete!")
    print("="*60)
    print(f"\nTo use this data, update BASE_DIR in the notebook:")
    print(f'  BASE_DIR = r"{os.path.abspath(output_dir)}"')
    print(f"\nOr use relative path:")
    print(f'  BASE_DIR = r"./data"')
    print("\nFiles generated:")
    print(f"  - {feature_file}")
    print(f"  - {target_file_uniform}")
    print(f"  - {target_file_skewed}")
    
    return feature_file, target_file_uniform, target_file_skewed


def main():
    parser = argparse.ArgumentParser(description="Generate sample data for federated learning")
    parser.add_argument("--num_samples", type=int, default=10000,
                       help="Number of data samples (default: 10000)")
    parser.add_argument("--num_clients", type=int, default=5,
                       help="Number of clients (default: 5)")
    parser.add_argument("--feature_dim", type=int, default=128,
                       help="Feature dimension (default: 128 for SIFT)")
    parser.add_argument("--output_dir", type=str, default="./data",
                       help="Output directory (default: ./data)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    generate_sample_data(
        num_samples=args.num_samples,
        num_clients=args.num_clients,
        feature_dim=args.feature_dim,
        output_dir=args.output_dir,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

