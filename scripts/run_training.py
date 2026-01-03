"""
Quick script to run federated learning training.

This script provides a command-line interface to run the federated learning
framework. For full functionality, use the Jupyter notebook.

Usage:
    python run_training.py --data_dir ./data --num_clients 10 --rounds 10
"""

import argparse
import sys
import os

# Add project directory to path
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_dir)

def main():
    parser = argparse.ArgumentParser(description="Run blockchain federated learning")
    parser.add_argument("--data_dir", type=str, default="./data",
                       help="Directory containing data files")
    parser.add_argument("--num_clients", type=int, default=10,
                       help="Number of clients")
    parser.add_argument("--rounds", type=int, default=10,
                       help="Number of federated rounds")
    parser.add_argument("--local_epochs", type=int, default=3,
                       help="Local training epochs")
    parser.add_argument("--batch_size", type=int, default=128,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                       help="Learning rate")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Blockchain Federated Learning")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Number of clients: {args.num_clients}")
    print(f"  Federated rounds: {args.rounds}")
    print(f"  Local epochs: {args.local_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    
    print("\n" + "="*60)
    print("NOTE: This script is a placeholder.")
    print("For full functionality, please use the Jupyter notebook:")
    print("  jupyter notebook Reduced_Layers_MUltihead.ipynb")
    print("="*60)
    
    print("\nTo run training:")
    print("1. Open Reduced_Layers_MUltihead.ipynb in Jupyter")
    print("2. Update BASE_DIR to point to your data")
    print("3. Run main_blockchain_federated()")
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"\n⚠️  Warning: Data directory '{args.data_dir}' does not exist")
        print("   Generate sample data using:")
        print("   python scripts/generate_sample_data.py --output_dir ./data")
    
    # Check if notebook exists
    notebook_path = os.path.join(project_dir, "Reduced_Layers_MUltihead.ipynb")
    if os.path.exists(notebook_path):
        print(f"\n✓ Notebook found: {notebook_path}")
    else:
        print(f"\n⚠️  Notebook not found: {notebook_path}")


if __name__ == "__main__":
    main()

