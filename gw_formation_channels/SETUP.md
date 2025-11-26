# Environment Setup Guide

## Overview

This project uses a dedicated Conda environment named `gw_channels` with Python 3.10. All dependencies have been successfully installed.

## Quick Start

### Option 1: Using the Helper Script (Recommended)
```bash
cd /Users/josephrodriguez/ASTROTHESIS/gw_formation_channels
source activate_env.sh
```

### Option 2: Manual Activation
```bash
conda activate gw_channels
```

## Installed Packages

### Core Deep Learning
- **PyTorch 2.9.1** (with Apple Silicon MPS support)
- **torchvision 0.24.1**

### Scientific Computing
- **NumPy 2.2.6**
- **SciPy**
- **Pandas**
- **h5py** (for COMPAS data files)

### Astronomy
- **Astropy**

### Machine Learning
- **scikit-learn**

### Data Processing
- **PyYAML**
- **tqdm**

### Visualization
- **matplotlib**
- **seaborn**
- **tensorboard**

### Development Tools
- **pytest** (testing)
- **black** (code formatting)
- **flake8** (linting)
- **isort** (import sorting)

### Documentation
- **Sphinx**
- **sphinx-rtd-theme**
- **nbsphinx**

### Jupyter
- **jupyter**
- **ipykernel**
- **ipywidgets**

## Running Scripts

Once the environment is activated, you can run scripts directly:

```bash
# Generate COMPAS ensemble
python -m compas_ensemble.generate_ensemble

# Train the model
python train.py

# Run TensorBoard
tensorboard --logdir runs/
```

## Running Without Activating (Alternative)

If you prefer not to activate the environment, you can run scripts using the full Python path:

```bash
/opt/anaconda3/envs/gw_channels/bin/python train.py
/opt/anaconda3/envs/gw_channels/bin/python -m compas_ensemble.generate_ensemble
```

## Jupyter Notebook Setup

The environment is configured for Jupyter notebooks. To use it:

```bash
conda activate gw_channels
jupyter notebook
```

Or with JupyterLab:

```bash
conda activate gw_channels
jupyter lab
```

## Apple Silicon (M1/M2/M3) GPU Support

PyTorch is installed with native MPS (Metal Performance Shaders) support for Apple Silicon GPUs:

```python
import torch

# Check if MPS is available
print(f"MPS available: {torch.backends.mps.is_available()}")

# Use MPS device in your code
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
```

## Troubleshooting

### "conda: command not found"
Make sure Anaconda is properly installed and initialized:
```bash
# Add conda to your PATH (if needed)
export PATH="/opt/anaconda3/bin:$PATH"
```

### Import errors
Make sure you've activated the environment:
```bash
conda activate gw_channels
```

### Permission errors
If you encounter permission errors, try running with the environment's Python directly:
```bash
/opt/anaconda3/envs/gw_channels/bin/python your_script.py
```

## Deactivating the Environment

When you're done working, deactivate the environment:
```bash
conda deactivate
```

## Updating Packages

To update all packages to their latest versions:
```bash
conda activate gw_channels
pip install --upgrade -r requirements.txt
```

## Environment Information

- **Environment name:** gw_channels
- **Python version:** 3.10.19
- **Location:** `/opt/anaconda3/envs/gw_channels`
- **Created:** November 26, 2025

## COMPAS Integration

The COMPAS binary evolution code is available at:
```
/Users/josephrodriguez/ASTROTHESIS/COMPAS
```

The Python utilities are accessible from this environment and can be imported directly:
```python
import sys
sys.path.append('/Users/josephrodriguez/ASTROTHESIS/COMPAS')
# Now you can import COMPAS Python utilities
```

## Project Structure

```
gw_formation_channels/
├── compas_ensemble/        # COMPAS ensemble generation
├── configs/                # Configuration files
├── data/                   # Data loaders (GWTC-4)
├── falsification/          # Falsification tests
├── inference/              # SBI framework
├── models/                 # Physics-informed neural networks
├── utils/                  # Utility functions
├── notebooks/              # Jupyter notebooks
├── train.py               # Main training script
├── requirements.txt       # Package dependencies
├── activate_env.sh        # Environment activation helper
└── SETUP.md              # This file
```

## Next Steps

1. Activate the environment: `source activate_env.sh`
2. Explore the notebooks in `notebooks/`
3. Generate a COMPAS ensemble: `python -m compas_ensemble.generate_ensemble`
4. Train the model: `python train.py`
5. Monitor training: `tensorboard --logdir runs/`

For more information, see the main [README.md](README.md) and [QUICKSTART.md](QUICKSTART.md).

