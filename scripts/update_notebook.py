"""
Script to update the notebook with all features integration.

Run this script to automatically update Reduced_Layers_MUltihead.ipynb with:
1. Metrics logger integration
2. Adaptive threshold
3. All three metrics hooks
"""

import json
import sys
import os

def update_notebook(notebook_path):
    """Update notebook with all features"""
    
    # Read notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    changes_made = []
    
    # Update each cell
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        
        source = ''.join(cell.get('source', []))
        
        # Update 1: Add metrics_logger parameter
        if 'def federated_round_async(self, local_epochs: int = 1, batch_size: int = 128,' in source:
            if 'metrics_logger=None' not in source:
                new_source = source.replace(
                    'def federated_round_async(self, local_epochs: int = 1, batch_size: int = 128,\n                              lr: float = 0.001, max_staleness: int = 3) -> Dict:',
                    'def federated_round_async(self, local_epochs: int = 1, batch_size: int = 128,\n                              lr: float = 0.001, max_staleness: int = 3,\n                              metrics_logger=None) -> Dict:'
                )
                cell['source'] = new_source.split('\n')
                changes_made.append(f"Cell {i}: Added metrics_logger parameter")
        
        # Update 2: Add Hook B
        if 'accepted_client_ids = {cid for cid, _ in ranked_updates}' in source:
            if 'Hook B: Log Security Metrics' not in source:
                hook_b = """        # Hook B: Log Security Metrics (Post-Consensus)
        if metrics_logger is not None:
            poc_scores = {cid: poc_contributions[cid]['overall_score'] for cid in accepted_client_ids}
            metrics_logger.log_security_metrics(
                accepted_client_ids=list(accepted_client_ids),
                proof_of_personhood=self.proof_of_personhood,
                poc_scores=poc_scores
            )
        
"""
                # Insert after the print statement
                lines = source.split('\n')
                new_lines = []
                for j, line in enumerate(lines):
                    new_lines.append(line)
                    if 'print(f"PoC: {len(accepted_client_ids)}/{len(pop_verified_updates)} updates passed threshold")' in line:
                        # Add hook after this line
                        new_lines.extend(hook_b.split('\n'))
                cell['source'] = new_lines
                changes_made.append(f"Cell {i}: Added Hook B (Security Metrics)")
        
        # Update 3: Add Hook C and Hook A
        if '# mine block and aggregate' in source and 'if self.blockchain.pending_transactions:' in source:
            if 'Hook C: Log Consensus Metrics' not in source:
                # This is complex, we'll need to do a more careful replacement
                hook_c = """            # Hook C: Log Consensus Metrics (During Block Creation)
            if metrics_logger is not None:
                committee_members = self.committee.committee_members
                staleness_values = [u[0].get("staleness", 0) for u in validated_updates]
                metrics_logger.log_consensus_metrics(
                    committee_members=committee_members,
                    accepted_updates=[u[0] for u in validated_updates],
                    staleness_values=staleness_values
                )
            
"""
                hook_a = """                # Hook A: Log Global Performance (Post-Aggregation)
                if metrics_logger is not None:
                    metrics_logger.log_global_performance(
                        model=self.global_model,
                        device=self.device,
                        criterion=torch.nn.MSELoss(),
                        batch_size=256
                    )
                
"""
                
                # Add Hook C before block_votes
                source = source.replace(
                    '            block_votes = {m: len(validated_updates) > 0 for m in self.committee.committee_members}',
                    hook_c + '            block_votes = {m: len(validated_updates) > 0 for m in self.committee.committee_members}'
                )
                
                # Add Hook A after _update_contributions_and_rewards
                source = source.replace(
                    '                self._update_contributions_and_rewards(validated_updates, validation_scores)\n                pruned =',
                    '                self._update_contributions_and_rewards(validated_updates, validation_scores)\n' + hook_a + '                pruned ='
                )
                
                cell['source'] = source.split('\n')
                changes_made.append(f"Cell {i}: Added Hook C and Hook A")
        
        # Update 4: Training loop - initialize metrics logger
        if 'X_train, X_val, Y_train, Y_val = train_test_split(' in source:
            if 'setup_test_data_for_metrics' not in source:
                metrics_init = """            # Setup test data for metrics logger
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
                # Add after print statement about train/val samples
                lines = source.split('\n')
                new_lines = []
                added = False
                for j, line in enumerate(lines):
                    new_lines.append(line)
                    if 'print(f"Train samples:' in line and not added:
                        new_lines.extend(metrics_init.split('\n'))
                        added = True
                cell['source'] = new_lines
                changes_made.append(f"Cell {i}: Added metrics logger initialization")
        
        # Update 5: Training loop - adaptive threshold
        if 'for round_num in range(1, FEDERATED_ROUNDS + 1):' in source:
            if 'get_adaptive_poc_threshold' not in source:
                # Replace the training loop
                old_loop = """            for round_num in range(1, FEDERATED_ROUNDS + 1):
                print(f"\\n--- Federated Round {round_num}/{FEDERATED_ROUNDS} ---")
                round_stats = fl_system.federated_round_async(
                    local_epochs=LOCAL_EPOCHS,
                    batch_size=BATCH_SIZE,
                    lr=LEARNING_RATE,
                    max_staleness=3
                )"""
                
                new_loop = """            # Use adaptive PoC threshold
            for round_num in range(1, FEDERATED_ROUNDS + 1):
                print(f"\\n--- Federated Round {round_num}/{FEDERATED_ROUNDS} ---")
                
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
                )"""
                
                source = source.replace(old_loop, new_loop)
                cell['source'] = source.split('\n')
                changes_made.append(f"Cell {i}: Added adaptive threshold and metrics logger to training loop")
        
        # Update 6: Add metrics generation after training
        if 'if __name__ == "__main__":' in source or 'main_blockchain_federated()' in source:
            if 'GENERATING METRICS AND PLOTS' not in source:
                # Find the end of training loop and add metrics generation
                metrics_gen = """
            # Generate metrics and plots (after training loop)
            if metrics_logger is not None:
                print("\\n" + "="*70)
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
                print("\\n" + "="*70)
                print("METRICS SUMMARY")
                print("="*70)
                print(f"Final Test Accuracy: {summary['final_test_accuracy']*100:.2f}%")
                print(f"Final Test Loss: {summary['final_test_loss']:.6f}")
                if summary.get('total_slashings', 0) > 0:
                    print(f"Total Slashings: {summary['total_slashings']}")
"""
                # Add before the end of the function or before if __name__
                lines = source.split('\n')
                # Find where to insert (before the last part of the function)
                # This is tricky, we'll add it before the closing of the try block or before if __name__
                new_lines = []
                inserted = False
                for j, line in enumerate(lines):
                    if ('if __name__' in line or 'main_blockchain_federated()' in line) and not inserted:
                        # Insert before this
                        new_lines.extend(metrics_gen.split('\n'))
                        inserted = True
                    new_lines.append(line)
                
                if not inserted:
                    # Add at the end before the last line
                    new_lines = lines[:-1] + metrics_gen.split('\n') + [lines[-1]]
                
                cell['source'] = new_lines
                changes_made.append(f"Cell {i}: Added metrics generation after training")
    
    # Write updated notebook
    if changes_made:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        
        print("="*70)
        print("NOTEBOOK UPDATED SUCCESSFULLY!")
        print("="*70)
        print(f"\nChanges made ({len(changes_made)}):")
        for change in changes_made:
            print(f"  ✓ {change}")
        print("\n✓ All features integrated!")
        print("  - Metrics logger parameter added")
        print("  - Hook A, B, C added")
        print("  - Adaptive threshold integrated")
        print("  - Metrics generation added")
    else:
        print("No changes needed - notebook may already be updated!")

if __name__ == "__main__":
    notebook_path = "Reduced_Layers_MUltihead.ipynb"
    if not os.path.exists(notebook_path):
        print(f"Error: {notebook_path} not found!")
        print("Make sure you're running from the project/ directory")
        sys.exit(1)
    
    update_notebook(notebook_path)

