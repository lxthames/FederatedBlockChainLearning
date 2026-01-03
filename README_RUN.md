# How to Run Experiments

## Quick Start

```bash
cd project
python scripts/run.py
```

## Prerequisites

### Option 1: Convert Notebook to Python (Recommended)

First, convert the notebook to a Python script:

```bash
cd project
jupyter nbconvert --to script Reduced_Layers_MUltihead.ipynb
```

This creates `Reduced_Layers_MUltihead.py` which can be imported.

### Option 2: Run from Notebook Context

If you want to run without converting, you can execute the script from within a notebook cell that has all the classes defined.

## Configuration

The script is configured with:
- **Client scales**: [4, 8]
- **Distribution**: ["skewed"]
- **Federated rounds**: 20
- **Local epochs**: 3
- **Batch size**: 128
- **Learning rate**: 0.001

To modify, edit the variables in `main()` function:
```python
CLIENT_SCALES = [4, 8]  # Change to [4] for quick test
DISTRIBUTIONS = ["skewed"]
FEDERATED_ROUNDS = 20  # Change to 5 for quick test
```

## Expected Output

```
======================================================================
BLOCKCHAIN-BASED DECENTRALIZED FEDERATED LEARNING (ENHANCED)
======================================================================
Device: cuda

======================================================================
BLOCKCHAIN FL | Clients: 4 | Distribution: SKEWED
======================================================================
Loaded X: (1000000, 128), Y: (1000000, 4)
Train samples: 900000, Val samples: 100000
✓ MetricsLogger initialized with test data
Distributed data to 4 clients
...

--- Federated Round 1/20 ---
PoC threshold: 0.0100
PoC: 3/4 updates passed threshold
Validated updates: 2/4
Block index: 1
Consensus achieved: True
...
```

## Output Files

After running, you'll get:
- **Models**: `data/4_blockchain_fl_model_skewed.pth`, `data/8_blockchain_fl_model_skewed.pth`
- **Metrics**: `data/metrics_4_skewed.json`, `data/metrics_8_skewed.json`
- **Plots**: `data/metrics_plots_4_skewed/`, `data/metrics_plots_8_skewed/`

## Troubleshooting

### "Could not import classes from notebook"

**Solution**: Convert notebook to .py first:
```bash
jupyter nbconvert --to script Reduced_Layers_MUltihead.ipynb
```

### "ModuleNotFoundError: No module named 'metrics_logger'"

**Solution**: Make sure you're in the `project/` directory and scripts folder exists.

### "File not found" errors

**Solution**: Check that data files exist in `data/` folder:
- `SIFT1B_train.npy`
- `gt_4M_client_counts_skewed.npy`
- `gt_8M_client_counts_skewed.npy`
- `SIFT_test_query.npy` (optional)

## Quick Test

For a quick test run, modify in `scripts/run.py`:
```python
CLIENT_SCALES = [4]  # Test with 4 clients only
FEDERATED_ROUNDS = 5  # Just 5 rounds
```

Then run:
```bash
python scripts/run.py
```

## Features Enabled

✅ Adaptive PoC threshold (fixes "0/X updates passed" issue)
✅ Metrics logging (performance, security, consensus)
✅ Test query data integration
✅ Automatic plot generation
✅ Model checkpointing

Happy experimenting! 🚀

