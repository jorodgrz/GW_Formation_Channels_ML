# ASTROTHESIS - Gravitational Wave Formation Channels Research

This repository contains research code for investigating gravitational wave formation channels using COMPAS (Compact Object Mergers: Population Astrophysics and Statistics) simulations combined with machine learning techniques.

## Project Overview

This project uses physics-informed neural networks and simulation-based inference to:
- Generate synthetic populations of binary compact objects using COMPAS
- Train machine learning models to identify formation channels
- Perform Bayesian inference on gravitational wave observations
- Test and falsify astrophysical models using GWTC (Gravitational Wave Transient Catalog) data

## Project Structure

```
ASTROTHESIS/
├── COMPAS/                    # COMPAS stellar evolution simulator
├── gw_formation_channels/     # Main project code
│   ├── compas_ensemble/       # COMPAS simulation generation
│   ├── data/                  # Data loaders (GWTC-4)
│   ├── models/                # Neural network architectures
│   ├── inference/             # SBI framework
│   ├── falsification/         # Model testing criteria
│   ├── configs/               # Configuration files
│   └── notebooks/             # Jupyter notebooks for analysis
├── my_results/                # Simulation outputs (ignored by git)
└── README.md                  # This file
```

## Key Features

- **COMPAS Integration**: Generate realistic populations of compact binary systems
- **Physics-Informed Neural Networks**: ML models constrained by physical laws
- **Simulation-Based Inference**: Bayesian parameter estimation using neural density estimators
- **Falsification Framework**: Rigorous statistical testing of astrophysical models
- **GWTC-4 Data Integration**: Analysis pipeline for gravitational wave observations

## Documentation

- [Quick Reference](gw_formation_channels/QUICKREF.md) - Command reference and common operations
- [Setup Guide](gw_formation_channels/SETUP.md) - Environment setup and installation
- [Quick Start](gw_formation_channels/QUICKSTART.md) - Get started quickly
- [Project Summary](gw_formation_channels/PROJECT_SUMMARY.md) - Detailed project overview
- [COMPAS Information](COMPAS_Info.md) - COMPAS simulator details

## Requirements

- Python 3.8+
- COMPAS simulator
- PyTorch
- h5py
- pandas, numpy, scipy
- sbi (simulation-based inference library)

## Installation

See [SETUP.md](gw_formation_channels/SETUP.md) for detailed installation instructions.

```bash
# Create conda environment
conda create -n gw_channels python=3.10
conda activate gw_channels

# Install dependencies
cd gw_formation_channels
pip install -r requirements.txt
```

## Usage

### Generate COMPAS Ensemble
```bash
python -m gw_formation_channels.compas_ensemble.generate_ensemble
```

### Train Neural Network
```bash
python gw_formation_channels/train.py
```

### Run Inference
```bash
python -m gw_formation_channels.inference.sbi_framework
```

## Citation

If you use this code in your research, please cite:
- COMPAS: [Stevenson et al. (2017)](https://arxiv.org/abs/1704.01352)
- This project: [To be published]

## License

This project is part of research conducted at UCSD Astronomy Club.

## Contact

For questions or collaboration inquiries, please open an issue on this repository.

