"""
Exact code snippets to update in the notebook

This file contains the exact code that needs to be added/updated in specific cells.
"""

# ============================================================================
# UPDATE 1: federated_round_async Method Signature (Cell ~17)
# ============================================================================
# Change from:
"""
    def federated_round_async(self, local_epochs: int = 1, batch_size: int = 128,
                              lr: float = 0.001, max_staleness: int = 3) -> Dict:
"""

# To:
"""
    def federated_round_async(self, local_epochs: int = 1, batch_size: int = 128,
                              lr: float = 0.001, max_staleness: int = 3,
                              metrics_logger=None) -> Dict:
"""

# ============================================================================
# UPDATE 2: Add Hook B - Security Metrics (After PoC filtering)
# ============================================================================
# Add after: accepted_client_ids = {cid for cid, _ in ranked_updates}
# Add before: # Step 4: Committee validation

"""
        accepted_client_ids = {cid for cid, _ in ranked_updates}
        print(f"PoC: {len(accepted_client_ids)}/{len(pop_verified_updates)} updates passed threshold")
        
        # Hook B: Log Security Metrics (Post-Consensus)
        if metrics_logger is not None:
            poc_scores = {cid: poc_contributions[cid]['overall_score'] for cid in accepted_client_ids}
            metrics_logger.log_security_metrics(
                accepted_client_ids=list(accepted_client_ids),
                proof_of_personhood=self.proof_of_personhood,
                poc_scores=poc_scores
            )
        
        # Step 4: Committee validation + scoring (only for PoC-accepted updates)
"""

# ============================================================================
# UPDATE 3: Add Hook C and Hook A (Block creation and aggregation)
# ============================================================================
# Replace the block mining section with:

"""
        # mine block and aggregate
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
                        criterion=torch.nn.MSELoss(),
                        batch_size=256
                    )
                
                pruned = self.storage_optimizer.cleanup_old_blocks(self.blockchain)
                if pruned:
                    print(f"Prunable old blocks (local) ≈ {pruned}")
"""

# ============================================================================
# UPDATE 4: Training Loop (Cell ~30, in main_blockchain_federated)
# ============================================================================
# Add before the training loop (after data splitting):

"""
            # Setup test data for metrics logger
            test_data = setup_test_data_for_metrics(TEST_QUERY_DATA, X_val, Y_val, num_clients)
            
            # Initialize metrics logger
            try:
                metrics_logger = MetricsLogger(
                    num_clients=num_clients,
                    test_data=test_data
                )
                print("✓ MetricsLogger initialized with test data")
            except NameError:
                metrics_logger = None
                print("⚠️ MetricsLogger not available (scripts not imported)")
"""

# Replace the training loop with:

"""
            # Use adaptive PoC threshold
            for round_num in range(1, FEDERATED_ROUNDS + 1):
                print(f"\n--- Federated Round {round_num}/{FEDERATED_ROUNDS} ---")
                
                # Set adaptive threshold (fixes PoC rejection issue)
                adaptive_threshold = get_adaptive_poc_threshold(round_num)
                fl_system.poc_threshold = adaptive_threshold
                print(f"PoC threshold: {adaptive_threshold:.4f}")
                
                # Start new round in metrics logger
                if metrics_logger is not None:
                    metrics_logger.start_new_round()
                
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

                val_loss = evaluate_global_model(fl_system.global_model, X_val, Y_val, device)
                print(f"Global model validation loss: {val_loss:.6f}")

                bc_info = fl_system.get_blockchain_info()
                print(f"Blockchain: {bc_info['chain_length']} blocks, Valid: {bc_info['is_valid']}, "
                      f"Version: {bc_info['current_version']}, Rewards: {bc_info['total_rewards']:.4f}")
                
                top_contributors = sorted(
                    fl_system.contribution_tracker.get_all_contributions().items(),
                    key=lambda x: x[1], reverse=True
                )[:3]
                print(f"Top contributors (id, contrib, reward): {[(cid, contrib, fl_system.incentive_mechanism.get_client_total_reward(cid)) for cid, contrib in top_contributors]}")
                
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
                    f"{BASE_DIR}/metrics_{num_clients}_{dist_type}.json"
                )
                
                metrics_logger.plot_all(
                    output_dir=f"{BASE_DIR}/metrics_plots_{num_clients}_{dist_type}",
                    show=False
                )
                
                summary = metrics_logger.get_summary()
                print("\n" + "="*70)
                print("METRICS SUMMARY")
                print("="*70)
                print(f"Final Test Accuracy: {summary['final_test_accuracy']*100:.2f}%")
                print(f"Final Test Loss: {summary['final_test_loss']:.6f}")
                if summary.get('total_slashings', 0) > 0:
                    print(f"Total Slashings: {summary['total_slashings']}")
"""

