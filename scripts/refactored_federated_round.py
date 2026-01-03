"""
Refactored federated_round_async Method

This shows the key changes needed in the federated_round_async method.
Replace the PoC evaluation section with this committee-based voting logic.
"""

def federated_round_async_refactored(self, local_epochs: int = 1, batch_size: int = 128,
                                     lr: float = 0.001, max_staleness: int = 3) -> Dict:
    """
    Refactored federated round with committee-based median voting for PoC.
    """
    current_version = self.model_version.get_current_version()
    client_updates, client_metrics = [], []

    # ========================================================================
    # Step 1: Local Training (unchanged)
    # ========================================================================
    for client in self.clients:
        cver = self.client_model_versions[client.client_id]
        staleness = current_version - cver
        gap = self.contribution_tracker.get_contribution_gap(client.client_id)
        if staleness > max_staleness or gap > 0.1:
            client.apply_global_update(self.global_model.state_dict())
            self.client_model_versions[client.client_id] = current_version
            cver = current_version
            staleness = 0
        loss = client.train_local(local_epochs, batch_size, lr)
        global_state = self.model_version.get_version(cver) or self.global_model.state_dict()
        update = client.get_model_update(global_state)
        update_norm = sum(torch.norm(v).item() for v in update.values() 
                         if isinstance(v, torch.Tensor) and v.dtype.is_floating_point)
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

    # ========================================================================
    # Step 2: PoP Verification (unchanged)
    # ========================================================================
    pop_verified_updates = []
    for update_info, metrics in zip(client_updates, client_metrics):
        client_id = update_info["client_id"]
        if self.proof_of_personhood.can_participate(client_id):
            pop_verified_updates.append((update_info, metrics))
        else:
            print(f"⚠️ Client {client_id} rejected: not PoP verified")

    # ========================================================================
    # Step 3: Committee-Based PoC Evaluation (NEW - REPLACES OLD PoC)
    # ========================================================================
    poc_scores = {}
    committee_members = self.committee.committee_members
    
    print(f"Committee members for voting: {committee_members}")
    
    for update_info, metrics in pop_verified_updates:
        client_id = update_info["client_id"]
        update = update_info["update"]
        
        # Get candidate model state (local model with update applied)
        candidate_model = copy.deepcopy(self.clients[client_id].model)
        candidate_state = candidate_model.state_dict()
        
        # Apply update to candidate state: w_candidate = w_local = w_global + delta
        # Since update = w_local - w_global, we have: w_candidate = w_global + update
        global_state = self.global_model.state_dict()
        for key in candidate_state:
            if key in update and isinstance(candidate_state[key], torch.Tensor):
                if key in global_state:
                    candidate_state[key] = global_state[key] + update[key]
        
        # Collect votes from committee members
        votes_loss = []
        votes_acc = []
        
        for committee_member_id in committee_members:
            # Committee members must be PoP verified
            if not self.proof_of_personhood.can_participate(committee_member_id):
                continue
            
            try:
                # Committee member validates the update on their local validation set
                committee_client = self.clients[committee_member_id]
                u_loss, u_acc = committee_client.validate_peer_update(
                    candidate_state, 
                    global_state, 
                    self.device
                )
                votes_loss.append(u_loss)
                votes_acc.append(u_acc)
            except Exception as e:
                print(f"⚠️ Committee member {committee_member_id} validation failed: {e}")
                continue
        
        # Calculate median of votes (robust to malicious members)
        if votes_loss and votes_acc:
            median_u_loss = np.median(votes_loss)
            median_u_acc = np.median(votes_acc)
            
            # Calculate robustness (global, not committee-based)
            u_rob = self.proof_of_contribution.calculate_robustness(update, LAMBDA)
            
            # Final PoC score using median formula
            # S_PoC(k) = gamma_1 * Median(u_loss) + gamma_2 * Median(u_acc) + gamma_3 * u_rob
            poc_score = (
                GAMMA_1 * median_u_loss +
                GAMMA_2 * median_u_acc +
                GAMMA_3 * u_rob
            )
            
            poc_scores[client_id] = {
                'u_loss': median_u_loss,
                'u_acc': median_u_acc,
                'u_rob': u_rob,
                'poc_score': poc_score,
                'votes_loss': votes_loss,
                'votes_acc': votes_acc,
                'num_votes': len(votes_loss)
            }
        else:
            # No valid votes
            poc_scores[client_id] = {
                'u_loss': 0.0,
                'u_acc': 0.0,
                'u_rob': 0.0,
                'poc_score': 0.0,
                'votes_loss': [],
                'votes_acc': [],
                'num_votes': 0
            }
    
    # ========================================================================
    # Step 4: Filter by Threshold (DELTA_TH)
    # ========================================================================
    accepted_updates = {
        cid: score for cid, score in poc_scores.items()
        if score['poc_score'] >= DELTA_TH
    }
    
    print(f"PoC: {len(accepted_updates)}/{len(pop_verified_updates)} updates passed threshold (DELTA_TH={DELTA_TH})")
    
    # Record contribution history
    for client_id, score in poc_scores.items():
        if client_id not in self.proof_of_contribution.contribution_history:
            self.proof_of_contribution.contribution_history[client_id] = []
        self.proof_of_contribution.contribution_history[client_id].append(score)
    
    # ========================================================================
    # Step 5: Committee Validation (optional additional layer)
    # ========================================================================
    validated_updates = []
    validation_scores = {}
    
    for update_info, metrics in pop_verified_updates:
        client_id = update_info["client_id"]
        
        # Only validate if passed PoC threshold
        if client_id not in accepted_updates:
            continue
        
        # Additional committee validation (if needed)
        votes, scores = {}, []
        for member_id in committee_members:
            if not self.proof_of_personhood.can_participate(member_id):
                continue
            
            accepted, score = self.committee.validate_update_with_score(
                update_info["update"], metrics
            )
            votes[member_id] = accepted
            if accepted:
                scores.append(score)
        
        accepted, _ = self.committee.aggregate_votes(votes)
        if accepted and scores:
            median_score = np.median(scores)  # Use median here too
            validation_scores[client_id] = median_score
            self.committee.record_validation_score(client_id, median_score)
            
            # Combine PoC score with committee validation score
            poc_score = poc_scores[client_id]['poc_score']
            combined_score = 0.6 * poc_score + 0.4 * median_score
            
            serialized_update = {
                k: {
                    "shape": list(v.shape) if isinstance(v, torch.Tensor) else [],
                    "norm": torch.norm(v).item() if isinstance(v, torch.Tensor) and v.dtype.is_floating_point else 0.0
                }
                for k, v in update_info["update"].items()
            }
            
            # Add PoC metrics to transaction
            serialized_update["poc_metrics"] = poc_scores[client_id]
            serialized_update["combined_score"] = combined_score
            
            # Sign transaction
            signature = self.identity_manager.sign(client_id, serialized_update)
            self.blockchain.add_transaction(
                client_id, serialized_update,
                local_epochs, update_info["data_size"], update_info["loss"],
                signature, self.identity_manager.public_ids[client_id]
            )
            validated_updates.append((update_info, metrics, combined_score))

    # ========================================================================
    # Step 6: Mine Block and Aggregate (unchanged)
    # ========================================================================
    if self.blockchain.pending_transactions:
        block_votes = {m: len(validated_updates) > 0 for m in committee_members}
        self.blockchain.mine_block(block_votes)
        if validated_updates:
            self.aggregate_updates_async(validated_updates)
            new_version = self.model_version.increment_version()
            self.model_version.store_version(new_version, self.global_model.state_dict())
            self.storage_optimizer.store_model(new_version, self.global_model.state_dict())
            self._update_contributions_and_rewards(validated_updates, validation_scores)
            pruned = self.storage_optimizer.cleanup_old_blocks(self.blockchain)
            if pruned:
                print(f"Prunable old blocks (local) ≈ {pruned}")

    return {
        "validated_updates": len(validated_updates),
        "total_updates": len(client_updates),
        "block_index": self.blockchain.get_latest_block().index,
        "consensus_achieved": len(validated_updates) > 0,
        "current_version": self.model_version.get_current_version()
    }

