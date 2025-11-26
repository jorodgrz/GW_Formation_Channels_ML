#!/bin/bash
# Helper script to activate the gw_channels conda environment
# Usage: source activate_env.sh

# Activate the conda environment
conda activate gw_channels

# Display environment information
echo "=========================================="
echo "GW Formation Channels Environment Active"
echo "=========================================="
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""
echo "Key packages installed:"
python -c "import numpy, torch, yaml, h5py, astropy, sklearn; print(f'  NumPy: {numpy.__version__}'); print(f'  PyTorch: {torch.__version__}'); print(f'  PyYAML: {yaml.__version__}'); print(f'  h5py: {h5py.__version__}')"
echo ""
echo "PyTorch MPS (Apple Silicon GPU) available: $(python -c 'import torch; print(torch.backends.mps.is_available())')"
echo ""
echo "=========================================="
echo "To deactivate: conda deactivate"
echo "=========================================="

