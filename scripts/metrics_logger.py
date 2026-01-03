"""
MetricsLogger: Centralized logging system for tracking Model Utility, Security, and Consensus Efficiency

Tracks metrics over T training rounds:
- Global Performance (test accuracy, test loss)
- Security (trust scores, attack success rate)
- Consensus Efficiency (committee participation, staleness)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
import json
import os


class MetricsLogger:
    """
    Centralized logger for federated learning metrics.
    Tracks utility, security, and consensus efficiency metrics.
    """
    
    def __init__(self, num_clients: int, test_data: Optional[Tuple] = None):
        """
        Initialize the metrics logger.
        
        Args:
            num_clients: Total number of clients in the system
            test_data: Optional (X_test, Y_test) for global model evaluation
        """
        self.num_clients = num_clients
        self.test_data = test_data
        
        # ====================================================================
        # Global Performance Metrics
        # ====================================================================
        self.test_accuracy_history: List[float] = []
        self.test_loss_history: List[float] = []
        
        # ====================================================================
        # Security Metrics (Trust & Attacks)
        # ====================================================================
        self.trust_scores_honest: List[float] = []  # Average τ for honest nodes per round
        self.trust_scores_sybil: List[float] = []  # Average τ for Sybil nodes per round
        self.attack_success_rate: List[float] = []  # ASR = Sybil Accepted / Sybil Proposed
        
        # Track Sybil updates per round
        self.sybil_updates_proposed: List[int] = []  # Sybil updates that passed threshold
        self.sybil_updates_accepted: List[int] = []  # Sybil updates included in block
        
        # Ground truth: which clients are malicious/Sybil
        self.sybil_clients: set = set()  # Will be set externally
        
        # ====================================================================
        # Consensus & PoC Metrics
        # ====================================================================
        self.committee_participation: Dict[int, int] = defaultdict(int)  # client_id -> count
        self.average_staleness: List[float] = []  # Average Δt_k for accepted updates
        
        # Additional tracking
        self.round_committee_members: List[List[int]] = []  # Committee members per round
        self.round_accepted_updates: List[List[int]] = []  # Accepted client IDs per round
        self.round_staleness_values: List[List[int]] = []  # Staleness values per round
        
        # Round counter
        self.current_round = 0
    
    def set_sybil_clients(self, sybil_client_ids: List[int]):
        """
        Set which clients are Sybil/malicious (ground truth).
        
        Args:
            sybil_client_ids: List of client IDs that are Sybil nodes
        """
        self.sybil_clients = set(sybil_client_ids)
    
    def log_global_performance(self, model, device, criterion=None, batch_size: int = 256):
        """
        Evaluate global model on test set and log accuracy/loss.
        Called after global model update (Equation 12).
        
        Args:
            model: Global model to evaluate
            device: Device for evaluation
            criterion: Loss function (default: MSELoss)
            batch_size: Batch size for evaluation
        """
        if self.test_data is None:
            # No test data available
            self.test_accuracy_history.append(0.0)
            self.test_loss_history.append(0.0)
            return
        
        X_test, Y_test = self.test_data
        
        if criterion is None:
            criterion = torch.nn.MSELoss()
        
        from torch.utils.data import DataLoader, TensorDataset
        
        test_dataset = TensorDataset(
            torch.from_numpy(X_test).float(),
            torch.from_numpy(Y_test)
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for Xb, Yb in test_loader:
                Xb = Xb.to(device)
                Yb = Yb.to(device)
                
                preds = model(Xb)
                loss = criterion(preds, Yb)
                total_loss += loss.item()
                
                # Calculate accuracy (for multi-output regression, use inverse MSE)
                mse = loss.item()
                # Simple accuracy proxy: 1 - normalized MSE
                batch_accuracy = max(0, 1.0 - mse / (torch.max(Yb).item() ** 2 + 1e-6))
                total_correct += batch_accuracy * len(Xb)
                total_samples += len(Xb)
        
        avg_loss = total_loss / len(test_loader)
        avg_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        self.test_loss_history.append(avg_loss)
        self.test_accuracy_history.append(avg_accuracy)
    
    def log_security_metrics(self, accepted_client_ids: List[int], 
                            proof_of_personhood, poc_scores: Dict[int, Dict]):
        """
        Log security metrics after consensus block is finalized.
        Called after P_accepted is determined.
        
        Args:
            accepted_client_ids: List of client IDs whose updates were accepted
            proof_of_personhood: PoP instance to get trust scores
            poc_scores: Dictionary of PoC scores {client_id: {poc_score, ...}}
        """
        # Identify Sybil and honest clients
        honest_clients = [cid for cid in range(self.num_clients) if cid not in self.sybil_clients]
        sybil_clients_list = [cid for cid in self.sybil_clients if cid < self.num_clients]
        
        # Calculate average trust scores
        honest_trust_scores = []
        sybil_trust_scores = []
        
        for client_id in range(self.num_clients):
            trust_score = proof_of_personhood.get_trust_score(client_id)
            if client_id in self.sybil_clients:
                sybil_trust_scores.append(trust_score)
            else:
                honest_trust_scores.append(trust_score)
        
        avg_honest_trust = np.mean(honest_trust_scores) if honest_trust_scores else 0.0
        avg_sybil_trust = np.mean(sybil_trust_scores) if sybil_trust_scores else 0.0
        
        self.trust_scores_honest.append(avg_honest_trust)
        self.trust_scores_sybil.append(avg_sybil_trust)
        
        # Calculate Attack Success Rate (ASR)
        # Count Sybil updates that passed threshold (proposed)
        sybil_proposed = 0
        for client_id in poc_scores.keys():
            if client_id in self.sybil_clients:
                sybil_proposed += 1
        
        # Count Sybil updates that were accepted (included in block)
        sybil_accepted = sum(1 for cid in accepted_client_ids if cid in self.sybil_clients)
        
        self.sybil_updates_proposed.append(sybil_proposed)
        self.sybil_updates_accepted.append(sybil_accepted)
        
        # Calculate ASR
        if sybil_proposed > 0:
            asr = sybil_accepted / sybil_proposed
        else:
            asr = 0.0  # No Sybil updates proposed
        
        self.attack_success_rate.append(asr)
    
    def log_consensus_metrics(self, committee_members: List[int], 
                            accepted_updates: List[Dict], staleness_values: List[int]):
        """
        Log consensus efficiency metrics during block creation.
        
        Args:
            committee_members: List of client IDs in the committee
            accepted_updates: List of update info dicts {client_id, ...}
            staleness_values: List of staleness values (Δt_k) for each accepted update
        """
        # Committee participation: increment count for each committee member
        for member_id in committee_members:
            self.committee_participation[member_id] += 1
        
        # Store round data
        self.round_committee_members.append(committee_members.copy())
        self.round_accepted_updates.append([u.get("client_id", -1) for u in accepted_updates])
        
        # Calculate average staleness
        if staleness_values:
            avg_staleness = np.mean(staleness_values)
        else:
            avg_staleness = 0.0
        
        self.average_staleness.append(avg_staleness)
        self.round_staleness_values.append(staleness_values.copy())
    
    def start_new_round(self):
        """Increment round counter"""
        self.current_round += 1
    
    def get_summary(self) -> Dict:
        """Get summary statistics of all metrics"""
        total_sybil_accepted = sum(self.sybil_updates_accepted)
        total_sybil_proposed = sum(self.sybil_updates_proposed)
        
        # Calculate overall ASR (only counting rounds with Sybil activity)
        if total_sybil_proposed > 0:
            overall_asr = total_sybil_accepted / total_sybil_proposed
        else:
            overall_asr = 0.0
        
        # Average ASR (mean of per-round ASR, including rounds with no activity)
        average_asr = np.mean(self.attack_success_rate) if self.attack_success_rate else 0.0
        
        return {
            "total_rounds": self.current_round,
            "final_test_accuracy": self.test_accuracy_history[-1] if self.test_accuracy_history else 0.0,
            "final_test_loss": self.test_loss_history[-1] if self.test_loss_history else 0.0,
            "final_honest_trust": self.trust_scores_honest[-1] if self.trust_scores_honest else 0.0,
            "final_sybil_trust": self.trust_scores_sybil[-1] if self.trust_scores_sybil else 0.0,
            "average_asr": average_asr,  # Mean of per-round ASR (includes rounds with 0 activity)
            "overall_asr": overall_asr,  # Total accepted / Total proposed (only rounds with activity)
            "total_sybil_accepted": total_sybil_accepted,
            "total_sybil_proposed": total_sybil_proposed,
            "rounds_with_sybil_activity": sum(1 for x in self.sybil_updates_proposed if x > 0),
            "average_staleness": np.mean(self.average_staleness) if self.average_staleness else 0.0,
            "committee_participation_counts": dict(self.committee_participation)
        }
    
    def save_metrics(self, filepath: str):
        """Save all metrics to JSON file"""
        metrics_dict = {
            "test_accuracy_history": self.test_accuracy_history,
            "test_loss_history": self.test_loss_history,
            "trust_scores_honest": self.trust_scores_honest,
            "trust_scores_sybil": self.trust_scores_sybil,
            "attack_success_rate": self.attack_success_rate,
            "sybil_updates_proposed": self.sybil_updates_proposed,
            "sybil_updates_accepted": self.sybil_updates_accepted,
            "average_staleness": self.average_staleness,
            "committee_participation": dict(self.committee_participation),
            "round_committee_members": self.round_committee_members,
            "round_accepted_updates": self.round_accepted_updates,
            "summary": self.get_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        print(f"✓ Metrics saved to {filepath}")
    
    def plot_global_convergence(self, save_path: Optional[str] = None, show: bool = True):
        """
        Plot 1: Global Convergence
        X-Axis: Communication Round (t)
        Y-Axis: Test Accuracy (%)
        """
        if not self.test_accuracy_history:
            print("⚠️ No test accuracy data to plot")
            return
        
        rounds = list(range(1, len(self.test_accuracy_history) + 1))
        accuracies = [acc * 100 for acc in self.test_accuracy_history]  # Convert to percentage
        
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, accuracies, 'b-', linewidth=2, marker='o', markersize=4)
        plt.xlabel('Communication Round (t)', fontsize=12)
        plt.ylabel('Test Accuracy (%)', fontsize=12)
        plt.title('Global Model Convergence: Test Accuracy Over Rounds', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, len(rounds) + 1)
        plt.ylim(0, max(accuracies) * 1.1 if accuracies else 100)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_trust_dynamics(self, save_path: Optional[str] = None, show: bool = True):
        """
        Plot 2: Trust Dynamics (The "Money Plot")
        X-Axis: Communication Round (t)
        Y-Axis: Average Trust Score (τ)
        Lines: Honest Nodes (→ 1.0) and Sybil Nodes (→ 0 or τ_min)
        """
        if not self.trust_scores_honest or not self.trust_scores_sybil:
            print("⚠️ No trust score data to plot")
            return
        
        rounds = list(range(1, len(self.trust_scores_honest) + 1))
        
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, self.trust_scores_honest, 'g-', linewidth=2, marker='o', 
                markersize=4, label='Honest Nodes', alpha=0.8)
        plt.plot(rounds, self.trust_scores_sybil, 'r-', linewidth=2, marker='s', 
                markersize=4, label='Sybil Nodes', alpha=0.8)
        
        plt.xlabel('Communication Round (t)', fontsize=12)
        plt.ylabel('Average Trust Score (τ)', fontsize=12)
        plt.title('Trust Dynamics: Honest vs Sybil Nodes', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11, loc='best')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, len(rounds) + 1)
        plt.ylim(0, 1.1)
        
        # Add reference lines
        plt.axhline(y=1.0, color='g', linestyle='--', alpha=0.3, label='Target (Honest)')
        plt.axhline(y=0.0, color='r', linestyle='--', alpha=0.3, label='Target (Sybil)')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_committee_fairness(self, save_path: Optional[str] = None, show: bool = True):
        """
        Plot 3: Committee Fairness Histogram
        X-Axis: Client IDs
        Y-Axis: Number of Selections
        Purpose: Demonstrate election probability prevents centralization
        """
        if not self.committee_participation:
            print("⚠️ No committee participation data to plot")
            return
        
        client_ids = sorted(self.committee_participation.keys())
        selection_counts = [self.committee_participation[cid] for cid in client_ids]
        
        # Color bars: green for honest, red for Sybil
        colors = ['red' if cid in self.sybil_clients else 'green' for cid in client_ids]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(client_ids, selection_counts, color=colors, alpha=0.7, edgecolor='black')
        
        plt.xlabel('Client ID', fontsize=12)
        plt.ylabel('Number of Committee Selections', fontsize=12)
        plt.title('Committee Fairness: Election Distribution Across Clients', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.xticks(client_ids)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='Honest Nodes'),
            Patch(facecolor='red', alpha=0.7, label='Sybil Nodes')
        ]
        plt.legend(handles=legend_elements, fontsize=11)
        
        # Add statistics text
        mean_selections = np.mean(selection_counts)
        std_selections = np.std(selection_counts)
        plt.text(0.02, 0.98, f'Mean: {mean_selections:.2f}\nStd: {std_selections:.2f}', 
                transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_all(self, output_dir: str = "./metrics_plots", show: bool = False):
        """
        Generate all standard plots and save them.
        
        Args:
            output_dir: Directory to save plots
            show: Whether to display plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        self.plot_global_convergence(
            save_path=os.path.join(output_dir, "global_convergence.png"),
            show=show
        )
        
        self.plot_trust_dynamics(
            save_path=os.path.join(output_dir, "trust_dynamics.png"),
            show=show
        )
        
        self.plot_committee_fairness(
            save_path=os.path.join(output_dir, "committee_fairness.png"),
            show=show
        )
        
        print(f"\n✓ All plots generated in {output_dir}/")

