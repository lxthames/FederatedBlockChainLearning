"""
Enhanced Blockchain Federated Learning

Main orchestration class that combines all components:
- PoP (Proof-of-Personhood)
- PoC (Proof-of-Contribution)
- Committee Consensus
- Blockchain
- Asynchronous FL
- Metrics Logging
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional

from .models import EnhancedMultiHeadDNN
from .blockchain import SignedBlockchain
from .consensus import EnhancedCommitteeConsensus
from .proof_of_personhood import ProofOfPersonhood
from .proof_of_contribution import ProofOfContribution
from .client import FederatedClient
from .identity import IdentityManager
from .persistence import ChainPersistence
from .utilities import (
    ModelVersion,
    ClientContributionTracker,
    StalenessHandler,
    IncentiveMechanism,
    StorageOptimizer
)


class EnhancedBlockchainFederatedLearning:
    """
    Enhanced blockchain-based FL with PoP, PoC, signatures, persistence, 
    async, and committee rotation.
    """
    
    def __init__(self, num_clients: int, committee_size: int, input_size: int,
                 num_outputs: int, device: torch.device,
                 election_method: str = "score", enable_async: bool = True,
                 persistence_path: str = "blockchain_state.json",
                 validation_data: Optional[Tuple] = None,
                 poc_threshold: float = 0.1,
                 top_k_updates: Optional[int] = None,
                 trust_threshold: float = 0.45):
        """
        Initialize enhanced blockchain federated learning system.
        
        Args:
            num_clients: Number of clients in the system
            committee_size: Size of the committee
            input_size: Input feature size
            num_outputs: Number of output heads (clients)
            device: Device to run on (cuda/cpu)
            election_method: Committee election method ("random", "score", "multi_factor")
            enable_async: Enable asynchronous federated learning
            persistence_path: Path to save blockchain state
            validation_data: Optional validation data tuple (X_val, Y_val)
            poc_threshold: Threshold for PoC acceptance
            top_k_updates: Optional limit on number of top updates to accept
            trust_threshold: Minimum trust score required for update acceptance (default: 0.45)
        """
        self.num_clients = num_clients
        self.device = device
        self.enable_async = enable_async
        self.poc_threshold = poc_threshold
        self.top_k_updates = top_k_updates
        self.trust_threshold = trust_threshold
        self.sybil_clients: List[int] = []
        self.attack_type: str = "none"

        # Identity & persistence
        self.identity_manager = IdentityManager(num_clients)
        self.persistence = ChainPersistence(persistence_path)

        # PoP & PoC mechanisms
        # Note: min_trust_score is for can_participate() check (registration requirement)
        # trust_threshold is separate and checked before PoC (prevention mechanism)
        self.proof_of_personhood = ProofOfPersonhood(min_trust_score=0.3)  # Lowered to allow initial participation
        self.proof_of_contribution = ProofOfContribution(validation_data=validation_data)

        # Core components
        self.blockchain = SignedBlockchain(self.persistence)
        self.committee = EnhancedCommitteeConsensus(committee_size, num_clients, election_method)
        self.blockchain.committee_members = self.committee.committee_members

        # Enhanced features
        self.model_version = ModelVersion()
        self.contribution_tracker = ClientContributionTracker(num_clients)
        self.staleness_handler = StalenessHandler(staleness_decay=0.9)
        self.incentive_mechanism = IncentiveMechanism(base_reward=1.0)
        self.storage_optimizer = StorageOptimizer(keep_latest_n_models=10)

        # Models
        self.global_model = EnhancedMultiHeadDNN(input_size, num_outputs).to(device)
        self.clients = []
        self.client_model_versions = {i: 0 for i in range(num_clients)}

    def set_sybil_config(self, sybil_clients: List[int], attack_type: str = "sign_flip"):
        """Configure Sybil clients and attack behavior."""
        self.sybil_clients = sybil_clients or []
        self.attack_type = attack_type

    def initialize_clients(self, client_data_list: List[Tuple]):
        """
        Initialize clients and register them with PoP.
        
        Args:
            client_data_list: List of (X_train, Y_train) tuples for each client
        """
        self.clients = []
        total_data_size = sum(len(X_train) for X_train, _ in client_data_list)
        
        for i, (X_train, Y_train) in enumerate(client_data_list):
            # Register node with PoP (challenge-based verification)
            success, token = self.proof_of_personhood.register_node(i, verification_method="challenge")
            if not success:
                print(f"⚠️ Client {i} PoP registration failed: {token}")
                continue
            
            client_model = copy.deepcopy(self.global_model)
            client = FederatedClient(i, client_model, (X_train, Y_train), self.device)
            self.clients.append(client)
            self.contribution_tracker.update_sample_share(i, len(X_train), total_data_size)
            print(f"✓ Client {i} registered with PoP (token: {token[:8]}...)")
        
        # Initial version
        v = self.model_version.increment_version()
        self.model_version.store_version(v, self.global_model.state_dict())
        self.storage_optimizer.store_model(v, self.global_model.state_dict())

    def federated_round_async(self, local_epochs: int = 1, batch_size: int = 128,
                              lr: float = 0.001, max_staleness: int = 3,
                              metrics_logger=None) -> Dict:
        """
        Execute one federated learning round with asynchronous updates.
        
        Args:
            local_epochs: Number of local training epochs
            batch_size: Batch size for training
            lr: Learning rate
            max_staleness: Maximum allowed model staleness
            metrics_logger: Optional MetricsLogger instance
            
        Returns:
            Dictionary with round statistics
        """
        current_version = self.model_version.get_current_version()
        client_updates, client_metrics = [], []

        # Local training (potentially stale models)
        for client in self.clients:
            cver = self.client_model_versions[client.client_id]
            staleness = current_version - cver
            gap = self.contribution_tracker.get_contribution_gap(client.client_id)
            
            # Update stale models
            if staleness > max_staleness or gap > 0.1:
                client.apply_global_update(self.global_model.state_dict())
                self.client_model_versions[client.client_id] = current_version
                cver = current_version
                staleness = 0
            
            # Train locally
            loss = client.train_local(local_epochs, batch_size, lr)
            global_state = self.model_version.get_version(cver) or self.global_model.state_dict()
            update = client.get_model_update(global_state)
            # Apply adversarial attack if this is a Sybil client
            if client.client_id in self.sybil_clients:
                update = self._apply_attack(update)
            update_norm = sum(
                torch.norm(v).item() 
                for v in update.values() 
                if isinstance(v, torch.Tensor) and v.dtype.is_floating_point
            )
            
            client_updates.append({
                "client_id": client.client_id,
                "update": update,
                "data_size": client.data_size,
                "loss": loss,
                "model_version": cver,
                "staleness": staleness
            })
            client_metrics.append({
                "client_id": client.client_id,
                "loss": loss,
                "update_norm": update_norm,
                "staleness": staleness
            })

        # Step 1: PoP verification - only registered nodes can participate
        # Note: can_participate() checks registration + basic trust (min_trust_score=0.3)
        # The stricter trust_threshold (0.45) is checked in Step 1.5 before PoC
        pop_verified_updates = []
        for update_info, metrics in zip(client_updates, client_metrics):
            client_id = update_info["client_id"]
            # Check if registered (basic requirement)
            if self.proof_of_personhood.verify_node(client_id):
                pop_verified_updates.append((update_info, metrics))
            else:
                print(f"⚠️ Client {client_id} rejected: not PoP registered")
        
        # Step 1.5: Trust threshold filter BEFORE PoC evaluation (Bootstrapping with Probation)
        # New nodes start with trust=0.5 (above threshold=0.45) to allow first participation
        # This enables validation of their contributions before penalizing
        trust_filtered_updates = []
        trust_rejected_early = []
        for update_info, metrics in pop_verified_updates:
            client_id = update_info["client_id"]
            trust_score = self.proof_of_personhood.get_trust_score(client_id)
            if trust_score >= self.trust_threshold:
                trust_filtered_updates.append((update_info, metrics))
            else:
                trust_rejected_early.append((client_id, trust_score))
                # NOTE: No trust penalty for administrative rejection (trust threshold)
                # Only penalize if update passes filters and is bad
        
        if trust_rejected_early:
            print(f"  🛡️ Trust filter: Rejected {len(trust_rejected_early)} updates before PoC (trust < {self.trust_threshold}): {[(cid, f'{score:.3f}') for cid, score in trust_rejected_early]} [No penalty - administrative rejection]")
        
        # Step 2: PoC evaluation - evaluate contribution of each update (only trust-filtered)
        poc_contributions = {}
        for update_info, metrics in trust_filtered_updates:
            client_id = update_info["client_id"]
            # Safety check: ensure client exists
            if client_id >= len(self.clients) or client_id < 0:
                print(f"⚠️ Client {client_id} out of range, skipping PoC evaluation")
                continue
            local_model = self.clients[client_id].model
            
            # Evaluate contribution using PoC
            contribution = self.proof_of_contribution.evaluate_contribution(
                client_id=client_id,
                model_update=update_info["update"],
                local_model=local_model,
                global_model=self.global_model,
                validation_data=self.proof_of_contribution.validation_data,
                device=self.device
            )
            poc_contributions[client_id] = contribution
        
        # Step 3: Filter by PoC threshold and rank
        threshold_met = {
            cid: contrib for cid, contrib in poc_contributions.items()
            if self.proof_of_contribution.meets_threshold(contrib['overall_score'], self.poc_threshold)
        }
        
        # Rank updates by contribution score
        ranked_updates = self.proof_of_contribution.rank_updates(
            threshold_met, 
            top_k=self.top_k_updates
        )
        
        accepted_client_ids = {cid for cid, _ in ranked_updates}
        print(f"PoC: {len(accepted_client_ids)}/{len(trust_filtered_updates)} updates passed threshold (trust-filtered: {len(trust_filtered_updates)}/{len(pop_verified_updates)})")
        
        # Note: Trust filtering already done in Step 1.5, so accepted_client_ids are already trust-filtered
        
        # Hook B: Log Security Metrics (Post-Consensus)
        if metrics_logger is not None:
            # Only include clients that have PoC contributions (safety check)
            poc_scores = {
                cid: poc_contributions[cid]['overall_score'] 
                for cid in accepted_client_ids 
                if cid in poc_contributions
            }
            metrics_logger.log_security_metrics(
                accepted_client_ids=list(accepted_client_ids),
                proof_of_personhood=self.proof_of_personhood,
                poc_scores=poc_scores
            )
        
        # Step 4: Committee validation + scoring (only for PoC-accepted AND trust-filtered updates)
        validated_updates = []
        validation_scores = {}
        for update_info, metrics in trust_filtered_updates:
            client_id = update_info["client_id"]
            
            # Only validate if passed PoC
            if client_id not in accepted_client_ids:
                continue
            
            votes, scores = {}, []
            for member_id in self.committee.committee_members:
                # Committee members must also be PoP verified to vote
                if not self.proof_of_personhood.can_participate(member_id):
                    continue
                
                accepted, score = self.committee.validate_update_with_score(update_info["update"], metrics)
                votes[member_id] = accepted
                if accepted:
                    scores.append(score)
            
            # Debug: Print committee validation details (only if rejected)
            if len(votes) == 0:
                print(f"  ⚠️ Client {client_id}: No committee members voted (committee_size={len(self.committee.committee_members)})")
            else:
                positive_votes = sum(1 for v in votes.values() if v)
                if positive_votes == 0:
                    # Only print if rejected to reduce noise
                    print(f"  ⚠️ Client {client_id}: Rejected (loss={metrics['loss']:.4f}, norm={metrics['update_norm']:.2f}, votes=0/{len(votes)})")
            
            accepted, consensus_ratio = self.committee.aggregate_votes(votes)
            if accepted and scores:
                median_score = np.median(scores)
                validation_scores[client_id] = median_score
                self.committee.record_validation_score(client_id, median_score)
                
                # Combine PoC score with committee validation score
                poc_score = poc_contributions[client_id]['overall_score']
                combined_score = 0.6 * poc_score + 0.4 * median_score
                
                # Serialize update (only float tensors)
                serialized_update = {
                    k: {
                        "shape": list(v.shape) if isinstance(v, torch.Tensor) else [],
                        "norm": torch.norm(v).item() if isinstance(v, torch.Tensor) and v.dtype.is_floating_point else 0.0
                    }
                    for k, v in update_info["update"].items()
                }
                
                # Add PoC metrics to transaction (convert numpy types to native Python types)
                poc_metrics = poc_contributions[client_id]
                serialized_update["poc_metrics"] = {
                    k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in poc_metrics.items()
                }
                serialized_update["combined_score"] = float(combined_score) if isinstance(combined_score, (np.floating, np.integer)) else combined_score
                
                # Sign transaction
                signature = self.identity_manager.sign(client_id, serialized_update)
                self.blockchain.add_transaction(
                    client_id, serialized_update,
                    local_epochs, update_info["data_size"], update_info["loss"],
                    signature, self.identity_manager.public_ids[client_id]
                )
                validated_updates.append((update_info, metrics, combined_score))
        
        # REPUTATION LOOP: Bootstrapping with Probation - Only penalize after validation
        # Key principle: Only penalize trust AFTER update passes all filters and is validated as bad
        # Do NOT penalize for administrative rejections (trust threshold, PoC threshold, committee)
        if hasattr(self, 'sybil_clients') and self.sybil_clients:
            # Track which clients we've already updated to avoid double-penalizing
            updated_clients = set()
            
            # Only update trust for updates that passed ALL filters (trust, PoC, committee)
            # These are the updates that were actually aggregated into the model
            for update_info, metrics, score in validated_updates:
                client_id = update_info["client_id"]
                if client_id in updated_clients:
                    continue  # Already updated
                updated_clients.add(client_id)
                
                if client_id in self.sybil_clients:
                    # Sybil update passed all filters and was accepted - STRONG PENALTY
                    # This means the system validated it as good, but it's actually malicious
                    current_trust = self.proof_of_personhood.get_trust_score(client_id)
                    new_trust = max(0.0, current_trust - 0.15)  # Strong penalty for validated bad update
                    self.proof_of_personhood.update_trust_score(client_id, new_trust)
                    # Remove probation status if in probation
                    if client_id in self.proof_of_personhood.registered_nodes:
                        self.proof_of_personhood.registered_nodes[client_id]["probation"] = False
                else:
                    # Honest update passed all filters and was accepted - REWARD
                    current_trust = self.proof_of_personhood.get_trust_score(client_id)
                    new_trust = min(1.0, current_trust + 0.02)  # Reward for good contribution
                    self.proof_of_personhood.update_trust_score(client_id, new_trust)
                    # Remove probation status after successful contribution
                    if client_id in self.proof_of_personhood.registered_nodes:
                        self.proof_of_personhood.registered_nodes[client_id]["probation"] = False
            
            # NOTE: We do NOT penalize clients whose updates were rejected by:
            # - Trust threshold (administrative rejection - no penalty)
            # - PoC threshold (quality filter - no penalty)
            # - Committee validation (consensus filter - no penalty)
            # Only penalize if update passes all filters and is bad (handled above)

        # Mine block and aggregate
        if self.blockchain.pending_transactions:
            # Hook C: Log Consensus Metrics (During Block Creation)
            if metrics_logger is not None:
                committee_members = self.committee.committee_members
                staleness_values = [u[0].get("staleness", 0) for u in validated_updates]
                metrics_logger.log_consensus_metrics(
                    committee_members=committee_members,
                    accepted_updates=[u[0] for u in validated_updates],
                    staleness_values=staleness_values
                )
            
            block_votes = {m: len(validated_updates) > 0 for m in self.committee.committee_members}
            self.blockchain.mine_block(block_votes)
            
            if validated_updates:
                self.aggregate_updates_async(validated_updates)
                new_version = self.model_version.increment_version()
                self.model_version.store_version(new_version, self.global_model.state_dict())
                self.storage_optimizer.store_model(new_version, self.global_model.state_dict())
                self._update_contributions_and_rewards(validated_updates, validation_scores)
                
                # Hook A: Log Global Performance (Post-Aggregation)
                if metrics_logger is not None:
                    metrics_logger.log_global_performance(
                        model=self.global_model,
                        device=self.device,
                        criterion=nn.MSELoss(),
                        batch_size=256
                    )
                
                pruned = self.storage_optimizer.cleanup_old_blocks(self.blockchain)
                if pruned:
                    print(f"Prunable old blocks (local) ≈ {pruned}")

        # Get block index safely
        block_index = 0
        if len(self.blockchain.chain) > 0:
            block_index = self.blockchain.get_latest_block().index
        
        return {
            "validated_updates": len(validated_updates),
            "total_updates": len(client_updates),
            "block_index": block_index,
            "consensus_achieved": len(validated_updates) > 0,
            "current_version": self.model_version.get_current_version()
        }

    def _apply_attack(self, update: Dict) -> Dict:
        """Apply the configured attack to a Sybil update."""
        attacked = {}
        for k, v in update.items():
            if not (isinstance(v, torch.Tensor) and v.dtype.is_floating_point):
                attacked[k] = v
                continue
            if self.attack_type in ("sign_flip", "label_flip"):
                attacked[k] = -v
            elif self.attack_type == "gaussian_noise":
                noise = 0.05 * torch.randn_like(v)
                attacked[k] = v + noise
            else:
                attacked[k] = v
        return attacked

    def aggregate_updates_async(self, validated_updates: List[Tuple]):
        """
        Aggregate validated updates with staleness-aware fusion weights.
        Includes trust-based weighting for prevention.
        
        Args:
            validated_updates: List of (update_info, metrics, score) tuples
        """
        total_data_size = sum(u[0]["data_size"] for u in validated_updates)
        if total_data_size == 0:
            # No updates to aggregate
            return
        current_version = self.model_version.get_current_version()
        fusion_weights = {}
        
        for update_info, metrics, score in validated_updates:
            cid = update_info["client_id"]
            base = update_info["data_size"] / total_data_size
            
            # PREVENTION: Trust-weighted aggregation (mitigates risk during probation)
            trust_score = self.proof_of_personhood.get_trust_score(cid)
            
            # Enhanced trust weighting: Use exponential scaling for stronger mitigation
            # trust=1.0 → weight=1.0 (100%)
            # trust=0.5 → weight=0.5 (50%) - probation/new nodes
            # trust=0.4 → weight=0.16 (16%) - low trust nodes contribute minimally
            # trust=0.3 → weight=0.09 (9%) - very low trust
            # This ensures low-trust nodes contribute minimally until they prove themselves
            trust_weight = max(0.0, min(1.0, trust_score ** 2))  # Quadratic scaling for stronger mitigation
            base *= trust_weight
            
            # Apply staleness penalty
            if self.enable_async:
                staleness = update_info.get("staleness", 0)
                base *= self.staleness_handler.calculate_staleness_penalty(
                    current_version - staleness, 
                    current_version
                )
            
            # Adjust for contribution gap
            gap = self.contribution_tracker.get_contribution_gap(cid)
            if gap > 0.05:
                base *= 1.0 + min(0.2, gap)
            
            fusion_weights[cid] = base
        
        # Normalize weights
        tw = sum(fusion_weights.values()) or 1.0
        fusion_weights = {k: v / tw for k, v in fusion_weights.items()}
        
        # Aggregate model weights
        global_state = self.global_model.state_dict()
        aggregated_state = {}
        
        for k in global_state:
            # Skip non-float tensors (like num_batches_tracked in BatchNorm)
            if isinstance(global_state[k], torch.Tensor) and not global_state[k].dtype.is_floating_point:
                # Copy integer tensors as-is (don't aggregate them)
                aggregated_state[k] = global_state[k].clone()
                continue
            
            ws = torch.zeros_like(global_state[k])
            for update_info, _, _ in validated_updates:
                cid = update_info["client_id"]
                weight = fusion_weights.get(cid, 0.0)
                client_state = self.clients[cid].model.state_dict()
                # Only aggregate float tensors
                if isinstance(client_state[k], torch.Tensor) and client_state[k].dtype.is_floating_point:
                    ws += weight * client_state[k]
                else:
                    # For non-float tensors, use the first client's value
                    if k not in aggregated_state:
                        aggregated_state[k] = client_state[k].clone()
            aggregated_state[k] = ws
        
        # Update global model
        self.global_model.load_state_dict(aggregated_state)
        
        # Update client models
        for update_info, _, _ in validated_updates:
            cid = update_info["client_id"]
            self.clients[cid].apply_global_update(aggregated_state)
            self.client_model_versions[cid] = current_version

    def _update_contributions_and_rewards(self, validated_updates: List[Tuple], 
                                         validation_scores: Dict[int, float]):
        """
        Update contribution tracking and distribute rewards.
        
        Args:
            validated_updates: List of validated updates
            validation_scores: Dictionary of validation scores
        """
        if not validated_updates:
            return  # No updates to process
        
        total_data_size = sum(update[0]["data_size"] for update in validated_updates)
        if total_data_size == 0:
            return  # Avoid division by zero
        
        client_rewards = {}
        
        for update_info, metrics, score in validated_updates:
            cid = update_info["client_id"]
            contrib = (update_info["data_size"] / total_data_size) * score
            self.contribution_tracker.record_contribution(cid, contrib)
            reward = self.incentive_mechanism.calculate_reward(cid, score, contrib)
            client_rewards[cid] = reward
        
        self.incentive_mechanism.distribute_rewards(client_rewards)

    def start_new_round(self):
        """Start a new training round and re-elect committee"""
        self.committee.start_new_round()
        self.blockchain.committee_members = self.committee.committee_members

    def get_blockchain_info(self) -> Dict:
        """Get blockchain information"""
        return {
            "chain_length": len(self.blockchain.chain),
            "pending_transactions": len(self.blockchain.pending_transactions),
            "committee_size": len(self.committee.committee_members),
            "is_valid": self.blockchain.validate_chain(),
            "current_version": self.model_version.get_current_version(),
            "total_rewards": self.incentive_mechanism.total_rewards_distributed
        }

    def get_client_statistics(self) -> Dict:
        """Get statistics for all clients"""
        stats = {}
        for cid in range(self.num_clients):
            stats[cid] = {
                **self.contribution_tracker.get_client_stats(cid),
                "total_reward": self.incentive_mechanism.get_client_total_reward(cid),
                "node_score": self.committee.node_scores.get(cid, 0.0),
                "model_version": self.client_model_versions.get(cid, 0)
            }
        return stats


# Alias for backward compatibility
BlockchainFederatedLearning = EnhancedBlockchainFederatedLearning

