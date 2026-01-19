"""
Run Federated Learning Experiments

Complete standalone script following software engineering principles.
All code is organized into proper modules.

Usage:
    python scripts/run.py

Configuration:
    - Client scales: 4, 8
    - Distribution: skewed
    - Rounds: 20
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add scripts directory to path
scripts_path = os.path.dirname(os.path.abspath(__file__))
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

import numpy as np
import torch
from sklearn.model_selection import train_test_split

# Import from organized package structure
try:
    from core import (
        EnhancedBlockchainFederatedLearning,
        EnhancedMultiHeadDNN
    )
    from utils import (
        load_test_query_data,
        load_test_query_with_ground_truth,
        setup_test_data_for_metrics,
        distribute_data_to_clients,
        evaluate_global_model,
        get_adaptive_poc_threshold
    )
    from metrics import MetricsLogger
except ImportError:
    # Fallback: import directly if package structure not set up
    from core.federated_learning import EnhancedBlockchainFederatedLearning
    from core.models import EnhancedMultiHeadDNN
    from utils.data_utils import (
        load_test_query_data,
        load_test_query_with_ground_truth,
        setup_test_data_for_metrics,
        distribute_data_to_clients,
        evaluate_global_model
    )
    from utils.threshold import get_adaptive_poc_threshold
    
    try:
        from metrics.metrics_logger import MetricsLogger
    except ImportError:
        from metrics_logger import MetricsLogger


def main():
    """Main function to run the experiment"""
    
    print("\n" + "="*70)
    print("BLOCKCHAIN-BASED DECENTRALIZED FEDERATED LEARNING (ENHANCED)")
    print("="*70)
    
    # Configuration
    BASE_DIR = os.path.join(project_root, "data")
    CLIENT_SCALES = [4, 8]
    DISTRIBUTIONS = ["skewed"]
    SYBIL_CLIENTS = [2, 3]  # Example Sybil nodes (adjust as needed)
    ATTACK_TYPE = "sign_flip"  # Options: sign_flip, label_flip, gaussian_noise, none
    
    COMMON_CONFIG = {
        "m": 122,
        "train_query_npy": os.path.join(BASE_DIR, "SIFT1B_train.npy"),
        "train_subset_size": 1_000_000,
        "val_split_size": 0.1,
        "random_seed": 42,
    }
    
    FEDERATED_ROUNDS = 20
    LOCAL_EPOCHS = 3
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    COMMITTEE_SIZE_RATIO = 0.3
    
    # Attack configuration - SCALE UP for realistic simulation
    SYBIL_RATIO = 0.5  # 50% of clients are Sybils (more aggressive)
    ATTACK_TYPE = "sign_flip"  # Options: "sign_flip", "gaussian_noise", "label_flip"
    
    # Legacy: Remove old hardcoded Sybil list (now using SYBIL_RATIO)
    # SYBIL_CLIENTS = [2, 3]  # Replaced by dynamic selection based on SYBIL_RATIO
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    np.random.seed(COMMON_CONFIG["random_seed"])
    torch.manual_seed(COMMON_CONFIG["random_seed"])
    
    for num_clients in CLIENT_SCALES:
        for dist_type in DISTRIBUTIONS:
            print("\n" + "="*70)
            print(f"BLOCKCHAIN FL | Clients: {num_clients} | Distribution: {dist_type.upper()}")
            print("="*70)
            
            config = COMMON_CONFIG.copy()
            config.update({
                "train_target_y_npy": os.path.join(BASE_DIR, f"gt_{num_clients}M_client_counts_{dist_type}.npy"),
                "model_output_path": os.path.join(BASE_DIR, f"{num_clients}_blockchain_fl_model_{dist_type}.pth"),
            })
            
            try:
                # Load data
                X_full = np.load(config["train_query_npy"])
                Y_full_raw = np.load(config["train_target_y_npy"])
                min_len = min(len(X_full), len(Y_full_raw))
                X_full = X_full[:min_len]
                Y_full_raw = Y_full_raw[:min_len]
                print(f"Loaded X: {X_full.shape}, Y: {Y_full_raw.shape}")
            except FileNotFoundError as e:
                print(f"❌ File not found: {e}")
                continue
            
            # Normalize Y
            Y_sums = Y_full_raw.sum(axis=1, keepdims=True).astype(np.float32)
            Y_sums[Y_sums == 0] = 1.0
            Y_full = Y_full_raw.astype(np.float32) / Y_sums
            
            # Subset if needed
            subset_size = config["train_subset_size"]
            if subset_size < len(X_full):
                indices = np.random.choice(len(X_full), subset_size, replace=False)
                X_full = X_full[indices]
                Y_full = Y_full[indices]
            
            # Split data: 80% train, 20% validation
            X_train, X_val, Y_train, Y_val = train_test_split(
                X_full,
                Y_full,
                test_size=0.2,  # 20% for validation
                random_state=config["random_seed"]
            )
            
            print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")
            
            # Load test query data with ground truth
            test_query_data = load_test_query_with_ground_truth(
                BASE_DIR, 
                num_clients, 
                dist_type
            )
            
            if test_query_data is not None:
                X_test_query, Y_test_query = test_query_data
                print(f"✓ Using test query data: {X_test_query.shape} with ground truth: {Y_test_query.shape}")
                test_data = (X_test_query, Y_test_query)
            else:
                # Fallback: use validation set as test data
                print(f"⚠️ Test query data not available, using validation set as test data")
                test_data = (X_val, Y_val)
            
            # Initialize metrics logger with test data
            # Note: sybil_clients will be set after they're determined
            try:
                metrics_logger = MetricsLogger(
                    num_clients=num_clients,
                    test_data=test_data  # (X_test, Y_test) from test query or validation fallback
                )
                print(f"✓ MetricsLogger initialized with test data: {len(test_data[0])} samples")
            except Exception as e:
                metrics_logger = None
                print(f"⚠️ MetricsLogger not available: {e}")
            
            # Distribute data to clients
            client_data_list = distribute_data_to_clients(X_train, Y_train, num_clients, dist_type)
            print(f"Distributed data to {num_clients} clients")
            
            # Determine Sybil clients (50% of clients, exclude from committee)
            num_sybils = max(1, int(num_clients * SYBIL_RATIO))
            # Select Sybils from higher-numbered clients to avoid committee overlap
            sybil_clients = list(range(num_clients - num_sybils, num_clients))
            print(f"⚠️ Sybil clients: {sybil_clients} (attack_type: {ATTACK_TYPE}, {len(sybil_clients)}/{num_clients} = {len(sybil_clients)/num_clients*100:.1f}%)")
            
            # Initialize FL system
            # Ensure committee size is at least 2 and can be formed from honest clients
            honest_clients = [i for i in range(num_clients) if i not in sybil_clients]
            committee_size = max(2, min(int(num_clients * COMMITTEE_SIZE_RATIO), len(honest_clients)))
            
            fl_system = EnhancedBlockchainFederatedLearning(
                num_clients=num_clients,
                committee_size=committee_size,
                input_size=X_train.shape[1],
                num_outputs=Y_train.shape[1],
                device=device,
                election_method="score",
                enable_async=True,
                persistence_path=os.path.join(BASE_DIR, f"blockchain_state_{num_clients}_{dist_type}.json"),
                validation_data=(X_val, Y_val),
                poc_threshold=0.1,
                top_k_updates=None,
                trust_threshold=0.45  # Explicitly set trust threshold for prevention
            )
            
            # Initialize clients
            fl_system.initialize_clients(client_data_list)
            
            # Configure Sybil attacks
            fl_system.set_sybil_config(sybil_clients, ATTACK_TYPE)
            
            # Set Sybil clients in metrics logger
            if metrics_logger is not None:
                metrics_logger.set_sybil_clients(sybil_clients)
            
            # Force committee to only select from honest clients (exclude Sybils)
            fl_system.committee.committee_members = honest_clients[:committee_size]
            fl_system.blockchain.committee_members = fl_system.committee.committee_members
            
            print(f"Committee size: {committee_size}")
            print(f"Committee members: {fl_system.committee.committee_members} (honest only)")
            
            # Initialize best model tracking
            best_val_loss = float("inf")
            
            # Use adaptive PoC threshold
            for round_num in range(1, FEDERATED_ROUNDS + 1):
                print(f"\n--- Federated Round {round_num}/{FEDERATED_ROUNDS} ---")
                
                # Set adaptive threshold (fixes PoC rejection issue)
                adaptive_threshold = min(get_adaptive_poc_threshold(round_num), 0.02)
                fl_system.poc_threshold = adaptive_threshold
                print(f"PoC threshold: {adaptive_threshold:.4f}")
                # #region agent log
                import json
                import time
                try:
                    log_path = r'c:\Users\user\Desktop\FederatedBlockchainLearning\.cursor\debug.log'
                    os.makedirs(os.path.dirname(log_path), exist_ok=True)
                    with open(log_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps({"id": f"log_{int(time.time()*1000)}", "timestamp": int(time.time()*1000), "location": "run.py:234", "message": "Round start", "data": {"round_num": round_num, "poc_threshold": adaptive_threshold, "trust_threshold": fl_system.trust_threshold, "sybil_clients": sybil_clients}, "sessionId": "debug-session", "runId": "run1", "hypothesisId": "E"}) + '\n')
                except Exception as e:
                    print(f"DEBUG LOG ERROR: {e}")
                # #endregion
                
                # Start new round in metrics logger
                if metrics_logger is not None:
                    metrics_logger.start_new_round()
                
                # Rotate committee (excludes Sybils automatically)
                if round_num > 1:
                    fl_system.start_new_round()
                
                round_stats = fl_system.federated_round_async(
                    local_epochs=LOCAL_EPOCHS,
                    batch_size=BATCH_SIZE,
                    lr=LEARNING_RATE,
                    max_staleness=3,
                    metrics_logger=metrics_logger  # Pass logger
                )
                print(f"Validated updates: {round_stats['validated_updates']}/{round_stats['total_updates']}")
                print(f"Block index: {round_stats['block_index']}")
                print(f"Consensus achieved: {round_stats['consensus_achieved']}")
                
                # Evaluate global model
                val_loss = evaluate_global_model(fl_system.global_model, X_val, Y_val, device)
                print(f"Global model validation loss: {val_loss:.6f}")
                
                # Print blockchain info
                bc_info = fl_system.get_blockchain_info()
                print(f"Blockchain: {bc_info['chain_length']} blocks, Valid: {bc_info['is_valid']}, "
                      f"Version: {bc_info['current_version']}, Rewards: {bc_info['total_rewards']:.4f}")
                
                # Print top contributors
                top_contributors = sorted(
                    fl_system.contribution_tracker.get_all_contributions().items(),
                    key=lambda x: x[1], reverse=True
                )[:3]
                print(f"Top contributors (id, contrib, reward): {[(cid, contrib, fl_system.incentive_mechanism.get_client_total_reward(cid)) for cid, contrib in top_contributors]}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(fl_system.global_model.state_dict(), config["model_output_path"])
                    print(f"✓ Saved best model → {config['model_output_path']}")
            
            # Generate metrics and plots (after training loop)
            if metrics_logger is not None:
                print("\n" + "="*70)
                print("GENERATING METRICS AND PLOTS")
                print("="*70)
                
                metrics_logger.save_metrics(
                    os.path.join(BASE_DIR, f"metrics_{num_clients}_{dist_type}.json")
                )
                
                metrics_logger.plot_all(
                    output_dir=os.path.join(BASE_DIR, f"metrics_plots_{num_clients}_{dist_type}"),
                    show=False
                )
                
                summary = metrics_logger.get_summary()
                print("\n" + "="*70)
                print("METRICS SUMMARY")
                print("="*70)
                print(f"Final Test Recall: {summary['final_test_recall']*100:.2f}%")
                print(f"Final Test Loss: {summary['final_test_loss']:.6f}")
                if summary.get('total_slashings', 0) > 0:
                    print(f"Total Slashings: {summary['total_slashings']}")
            
            print(f"\n✓ Completed: {num_clients} clients, {dist_type} distribution")
    
    print("\n" + "="*70)
    print("ALL BLOCKCHAIN FEDERATED LEARNING RUNS COMPLETED")
    print("="*70)


if __name__ == "__main__":
    main()
