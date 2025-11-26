# Physics-Informed Deep Learning for GW Formation Channel Inference

A sophisticated framework for quantifying astrophysical uncertainties in gravitational wave formation channels using ensemble population synthesis codes as Bayesian priors.

**Institution:** UC San Diego 
**Project:** ML Gravitational Wave

**Research Question** Research Question: How can a physics-informed deep learning architecture use an ensemble of population synthesis codes (COMPAS, COSMIC, SEVN) as Bayesian priors to jointly perform simulation-based inference and domain adaptation on gravitational wave data, thereby quantifying both astrophysical model uncertainty (epistemic) and detector noise uncertainty (aleatoric) in formation-channel likelihoods? Specifically, the architecture will be falsified if: (1) ensemble epistemic uncertainty (mutual information across code predictions) exceeds observational uncertainty for >50% of GWTC-4 events, indicating that stellar evolution model systematics prevent meaningful channel inference regardless of domain adaptation; or (2) cross-modal attention weights fail to isolate common envelope efficiency (α_CE) as the primary driver of Channel I/IV degeneracy (rank correlation <0.5), contradicting the hypothesis that CE physics dominates formation-channel diversity.

---

## Overview

This project implements a cutting-edge approach combining:
- **Ensemble Population Synthesis**: Systematic variation of astrophysical parameters using COMPAS
- **Physics-Informed Deep Learning**: Neural networks that encode population synthesis outputs
- **Uncertainty Quantification**: Separates epistemic (model) and aleatoric (noise) uncertainties
- **Simulation-Based Inference**: Bayesian inference with learned priors from population synthesis
- **Falsification Testing**: Rigorous criteria for determining when inferences should be rejected

### Key Features

1. **Multi-Code Ensemble**: Combines COMPAS, COSMIC, and SEVN population synthesis codes
2. **Cross-Modal Attention**: Identifies which physics parameters (especially α_CE) drive formation channels
3. **Domain Adaptation**: Bridges the gap between simulated populations and real GW observations
4. **Uncertainty Decomposition**: Quantifies both reducible (epistemic) and irreducible (aleatoric) uncertainties
5. **Falsification Criteria**: Tests when formation channel claims should be rejected

---

## Formation Channels

The framework classifies BBH mergers into four primary formation channels:

- **Channel I**: Isolated binary evolution with stable mass transfer
- **Channel II**: Dynamical formation in dense stellar environments (globular clusters)
- **Channel III**: Hierarchical triple systems (Kozai-Lidov evolution)
- **Channel IV**: Common envelope evolution + stable mass transfer

---

## Installation

### Prerequisites

- Python 3.8+
- COMPAS v02.41.04+ (already installed at `/Users/josephrodriguez/ASTROTHESIS/COMPAS`)
- CUDA-capable GPU (optional, recommended for training)
- 16GB+ RAM (32GB+ recommended for large ensembles)

### Setup

```bash
# Navigate to project directory
cd /Users/josephrodriguez/ASTROTHESIS/gw_formation_channels

# Create conda environment (preferred)
conda create -n gw_channels python=3.10
conda activate gw_channels

# Install requirements
pip install -r requirements.txt

# Install COMPAS Python utilities
cd ../COMPAS
pip install -e .
cd ../gw_formation_channels
```

---

## Project Structure

```
gw_formation_channels/
├── compas_ensemble/          # COMPAS ensemble generation
│   ├── generate_ensemble.py  # Main ensemble generation script
│   └── __init__.py
│
├── models/                   # Neural network architectures
│   ├── physics_informed_nn.py # Main physics-informed model
│   └── __init__.py
│
├── inference/                # Simulation-based inference
│   ├── sbi_framework.py     # Neural posterior estimation
│   └── __init__.py
│
├── falsification/            # Falsification testing
│   ├── test_criteria.py     # Test falsification criteria
│   └── __init__.py
│
├── data/                     # Data loading utilities
│   ├── gwtc4_loader.py      # GWTC-4 data loader
│   └── __init__.py
│
├── utils/                    # Utility functions
│   └── __init__.py
│
├── configs/                  # Configuration files
│   └── default_config.yaml  # Default configuration
│
├── notebooks/                # Jupyter notebooks for analysis
│
├── train.py                  # Main training script
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

---

## Usage

### 1. Generate COMPAS Ensemble

First, generate a systematic grid of COMPAS simulations varying key astrophysical parameters:

```bash
# Test run (3 parameter combinations, fast)
python -m compas_ensemble.generate_ensemble \
    --compas-binary /Users/josephrodriguez/ASTROTHESIS/COMPAS/src/bin/COMPAS \
    --output-dir ./compas_ensemble_output \
    --n-systems 10000 \
    --test-run

# Production run (full grid, ~hours to days depending on grid size)
python -m compas_ensemble.generate_ensemble \
    --compas-binary /Users/josephrodriguez/ASTROTHESIS/COMPAS/src/bin/COMPAS \
    --output-dir ./compas_ensemble_output \
    --n-systems 100000 \
    --n-alpha-points 10
```

**Key Parameters Varied:**
- `α_CE`: Common envelope efficiency [0.1, 5.0] - **primary driver**
- `λ_CE`: CE lambda parameter [0.05, 1.0]
- Natal kicks: [150, 217, 300] km/s
- Metallicity: [0.0001, 0.0142, 0.02]
- Mass transfer efficiency: [0.25, 0.5, 0.75]

### 2. Download GWTC-4 Data

Download gravitational wave observations from the Gravitational Wave Transient Catalog:

```bash
# Download from GWOSC/Zenodo
# URL: https://zenodo.org/record/8177023/files/GWTC-4_posteriors.h5
# Save to: ./gwtc4_data/GWTC-4_posteriors.h5

# Or create synthetic test data
python -c "from data.gwtc4_loader import create_synthetic_gwtc4_for_testing; \
           create_synthetic_gwtc4_for_testing(n_events=20, output_file='./test_gwtc4.h5')"
```

### 3. Train the Model

Train the physics-informed ensemble model:

```bash
# Test mode with dummy data
python train.py --config configs/default_config.yaml --test-mode

# Production training (requires real data)
python train.py --config configs/default_config.yaml

# Resume from checkpoint
python train.py --config configs/default_config.yaml --checkpoint checkpoints/checkpoint_epoch_50.pth
```

**Training monitors:**
- TensorBoard: `tensorboard --logdir runs/`
- Checkpoints saved to: `checkpoints/`
- Logs saved to: `logs/`

### 4. Test Falsification Criteria

After training, test whether formation channel inferences should be rejected:

```python
from falsification.test_criteria import FalsificationTester
from data.gwtc4_loader import GWTC4Loader
import torch

# Load model
model = torch.load('checkpoints/best_model.pth')

# Load GWTC-4 catalog
loader = GWTC4Loader('./gwtc4_data/GWTC-4_posteriors.h5')
events = loader.load_all_events(max_events=None)

# Test falsification criteria
tester = FalsificationTester(
    model=model,
    gwtc4_catalog=events,
    output_dir='./falsification_results'
)

# Run all tests
results = tester.run_all_tests(
    criterion_1_threshold=0.5,
    criterion_2_min_corr=0.5,
    save_plots=True
)

print(f"Overall falsified: {results['overall_falsified']}")
```

**Falsification Criteria:**

1. **Criterion 1: Epistemic > Observational Uncertainty**
   - If epistemic uncertainty exceeds measurement uncertainty for >50% of events → **FALSIFIED**
   - Indicates models disagree too much for reliable inference

2. **Criterion 2: α_CE Rank Correlation**
   - If rank correlation between α_CE attention and Channel I/IV assignment < 0.5 → **FALSIFIED**
   - Indicates model fails to identify the key physics parameter

---

## Configuration

Edit `configs/default_config.yaml` to customize:

```yaml
# Model architecture
model:
  n_codes: 3  # COMPAS, COSMIC, SEVN
  latent_dim: 64
  n_channels: 4

# Training
training:
  batch_size: 256
  learning_rate: 0.001
  n_epochs: 100

# Falsification
falsification:
  criterion_1:
    threshold: 0.5  # 50% of events
  criterion_2:
    min_correlation: 0.5
```

---

## Results and Outputs

### Generated Outputs

1. **COMPAS Ensemble**: `compas_ensemble_output/`
   - HDF5 files with DCO populations for each parameter combination
   - Metadata: `ensemble_metadata.json`

2. **Model Checkpoints**: `checkpoints/`
   - `best_model.pth`: Best validation performance
   - `checkpoint_epoch_X.pth`: Periodic checkpoints

3. **Falsification Results**: `falsification_results/`
   - `falsification_summary.json`: Overall results
   - `criterion_1_results.csv`: Per-event Criterion 1 results
   - `criterion_2_results.csv`: α_CE correlation results
   - Diagnostic plots: `criterion_1_plots.png`, `criterion_2_plots.png`

4. **Logs**: `logs/` and `runs/` (TensorBoard)

### Interpreting Results

**If PASSED (not falsified):**
- Formation channel inferences are reliable
- Proceed with scientific interpretation
- Report uncertainties in publications

**If FALSIFIED:**
- Formation channel inferences should NOT be trusted
- Models disagree too much (Criterion 1) or
- Wrong physics parameter identified (Criterion 2)
- More work needed on population synthesis modeling

---

## Scientific Context

### Research Question

> Can we reliably infer the formation channel of observed gravitational wave events from their mass and spin properties, given uncertainties in stellar evolution physics?

### Key Challenges

1. **Channel Degeneracy**: Different formation channels can produce similar final systems
2. **Model Uncertainty**: Population synthesis codes make different assumptions (α_CE, kicks, mass transfer)
3. **Limited Observables**: We only observe final masses and spins, not full evolutionary history
4. **Data Scarcity**: ~100 events in GWTC-4, but many possible formation pathways

### Novel Approach

This framework addresses these challenges by:
- Using **ensemble methods** to quantify model uncertainty
- **Physics-informed priors** from population synthesis instead of uninformative priors
- **Explicit falsification criteria** to know when to reject claims
- **Uncertainty decomposition** to separate model vs. measurement limitations

---

## References

### Population Synthesis

- COMPAS: [Stevenson et al. 2017](https://ui.adsabs.harvard.edu/abs/2017NatCo...814906S)
- COSMIC: [Breivik et al. 2020](https://ui.adsabs.harvard.edu/abs/2020ApJ...898...71B)
- SEVN: [Spera et al. 2019](https://ui.adsabs.harvard.edu/abs/2019MNRAS.485..889S)

### Formation Channels

- Isolated Binary Evolution: [Belczynski et al. 2016](https://ui.adsabs.harvard.edu/abs/2016A%26A...594A..97B)
- Dynamical Assembly: [Rodriguez et al. 2016](https://ui.adsabs.harvard.edu/abs/2016PhRvD..93h4029R)
- Hierarchical Triples: [Antonini et al. 2017](https://ui.adsabs.harvard.edu/abs/2017ApJ...841...77A)

### GWTC-4

- LIGO/Virgo/KAGRA: [The LIGO Scientific Collaboration et al. 2021](https://ui.adsabs.harvard.edu/abs/2021arXiv211103606T)

### Uncertainty Quantification

- Simulation-Based Inference: [Cranmer et al. 2020](https://doi.org/10.1073/pnas.1912789117)
- Epistemic Uncertainty: [Kendall & Gal 2017](https://arxiv.org/abs/1703.04977)

---

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run specific test
pytest py_tests/test_models.py

# With coverage
pytest --cov=gw_formation_channels --cov-report=html
```

### Code Formatting

```bash
# Format code
black gw_formation_channels/

# Sort imports
isort gw_formation_channels/

# Lint
flake8 gw_formation_channels/
```

---

## Contributing

This project is part of the ASTROTHESIS research initiative at UCSD's Astronomy Club.

### Guidelines

1. Thoroughly comment all code (see memory preference)
2. Use descriptive variable names
3. Write tests for new features
4. Update documentation
5. Follow PEP 8 style guidelines

---

## License

This project is developed for academic research purposes at UCSD.

---

## Contact

**Project Lead:** Joseph Rodriguez  
**Institution:** Astronomy Club at UCSD  
**Project:** ASTROTHESIS

---

## Acknowledgments

- COMPAS development team
- LIGO/Virgo/KAGRA collaborations
- UCSD Astronomy Club
- All contributors to population synthesis and gravitational wave research

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{gw_formation_channels,
  author = {Rodriguez, Joseph},
  title = {Physics-Informed Deep Learning for GW Formation Channel Inference},
  year = {2025},
  institution = {Astronomy Club at UCSD},
  url = {https://github.com/UCSD-Astronomy/ASTROTHESIS}
}
```

---

**Last Updated:** November 2025  
**Version:** 0.1.0

