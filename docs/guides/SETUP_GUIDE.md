# Setup Guide

Complete guide for setting up the Blockchain-Based Decentralized Federated Learning Framework.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **Operating System**: Windows, macOS, or Linux
- **Python**: 3.8 or higher
- **RAM**: Minimum 4GB (8GB+ recommended)
- **Storage**: 1GB free space
- **GPU**: Optional but recommended for faster training

### Software Requirements

- Python 3.8+
- pip (Python package manager)
- Jupyter Notebook (for running the main notebook)
- Git (optional, for version control)

## Installation

### Step 1: Clone Repository

```bash
# If using Git
git clone <repository-url>
cd Federated_BL

# Or download and extract the ZIP file
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

#### Option A: Using requirements.txt

```bash
pip install -r requirements.txt
```

#### Option B: Manual Installation

```bash
# Core dependencies
pip install torch torchvision torchaudio
pip install numpy scikit-learn tqdm jupyter

# Optional: For GPU support (CUDA)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 4: Verify Installation

```python
# Test imports
python -c "import torch; import numpy; import sklearn; print('All dependencies installed!')"
```

## Configuration

### Data Configuration

1. **Prepare Data Files**:
   - Place SIFT feature files in `data/` directory
   - Ensure files are in `.npy` format
   - Update paths in the notebook if needed

2. **Data Paths**:
   ```python
   BASE_DIR = r"C:\Users\user\Downloads"  # Update this path
   TRAIN_FILE = "SIFT1B_train.npy"
   TARGET_FILES = "gt_*M_client_counts_*.npy"
   ```

### System Configuration

#### Device Selection

```python
# CPU only
device = torch.device("cpu")

# GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

#### Federated Learning Parameters

```python
# Number of clients
num_clients = 10

# Committee size (typically 20-30% of clients)
committee_size = max(1, int(num_clients * 0.25))

# Training parameters
LOCAL_EPOCHS = 3
BATCH_SIZE = 128
LEARNING_RATE = 0.001
FEDERATED_ROUNDS = 10
```

#### PoP/PoC Configuration

```python
# PoP settings
min_trust_score = 0.5  # Minimum trust to participate

# PoC settings
poc_threshold = 0.1   # Minimum contribution score
top_k_updates = None  # None = all above threshold, or set to N for top-N
```

### Blockchain Configuration

```python
# Persistence path
persistence_path = "blockchain_state.json"

# Committee election method
election_method = "score"  # Options: "score", "random", "multi_factor"
```

## Verification

### Test Installation

1. **Open Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

2. **Open Test Notebook**:
   - Open `Reduced_Layers_MUltihead.ipynb`
   - Run the first few cells to test imports

3. **Run Quick Test**:
   ```python
   # Test basic functionality
   import torch
   import numpy as np
   
   # Test model creation
   from Reduced_Layers_MUltihead import EnhancedMultiHeadDNN
   
   model = EnhancedMultiHeadDNN(input_size=128, num_clients=5)
   x = torch.randn(10, 128)
   y = model(x)
   print(f"Model output shape: {y.shape}")  # Should be (10, 5)
   ```

### Verify GPU (Optional)

```python
import torch

if torch.cuda.is_available():
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("GPU not available, using CPU")
```

## Troubleshooting

### Common Issues

#### Issue 1: Import Errors

**Problem**: `ModuleNotFoundError` when importing packages

**Solution**:
```bash
# Ensure virtual environment is activated
# Reinstall packages
pip install --upgrade -r requirements.txt
```

#### Issue 2: CUDA/GPU Issues

**Problem**: GPU not detected or CUDA errors

**Solution**:
```python
# Check CUDA availability
import torch
print(torch.cuda.is_available())

# If False, install CUDA-enabled PyTorch
# Or use CPU: device = torch.device("cpu")
```

#### Issue 3: Memory Errors

**Problem**: Out of memory errors during training

**Solution**:
- Reduce batch size: `BATCH_SIZE = 64` or `32`
- Reduce number of clients
- Use CPU instead of GPU
- Reduce data subset size

#### Issue 4: File Not Found

**Problem**: Data files not found

**Solution**:
- Check file paths in notebook
- Ensure data files are in correct directory
- Update `BASE_DIR` path

#### Issue 5: Blockchain Persistence Errors

**Problem**: Cannot save/load blockchain state

**Solution**:
- Check file permissions
- Ensure directory exists
- Check disk space
- Verify JSON format

### Getting Help

1. **Check Documentation**:
   - [Usage Guide](USAGE_GUIDE.md)
   - [API Reference](../api/API_REFERENCE.md)
   - [Architecture Docs](../architecture/ARCHITECTURE.md)

2. **Check Logs**:
   - Review console output
   - Check error messages
   - Inspect blockchain JSON files

3. **Report Issues**:
   - Open an issue on GitHub
   - Include error messages
   - Provide system information

## Next Steps

After setup:

1. Read [Usage Guide](USAGE_GUIDE.md) for how to use the system
2. Review [Architecture Documentation](../architecture/ARCHITECTURE.md) for system design
3. Check [PoP/PoC Guide](POP_POC_GUIDE.md) for security mechanisms
4. Run example training: `main_blockchain_federated()`

## Related Documentation

- [Usage Guide](USAGE_GUIDE.md)
- [Architecture Overview](../architecture/ARCHITECTURE.md)
- [API Reference](../api/API_REFERENCE.md)

