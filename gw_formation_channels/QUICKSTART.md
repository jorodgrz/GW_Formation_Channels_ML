# Quick Start Guide

Get started with physics-informed gravitational wave formation channel inference in 5 minutes.

## Installation

```bash
# Clone or navigate to project
cd /Users/josephrodriguez/ASTROTHESIS/gw_formation_channels

# Create environment
conda create -n gw_channels python=3.10
conda activate gw_channels

# Install
pip install -r requirements.txt
pip install -e .
```

## Run Test Example (5 minutes)

### 1. Generate Small Test Ensemble

```bash
python -m compas_ensemble.generate_ensemble \
    --compas-binary /Users/josephrodriguez/ASTROTHESIS/COMPAS/src/bin/COMPAS \
    --output-dir ./test_ensemble \
    --n-systems 1000 \
    --test-run
```

Expected output: 3 COMPAS runs with different parameters (takes ~2-3 minutes)

### 2. Create Synthetic GWTC-4 Data

```python
from data.gwtc4_loader import create_synthetic_gwtc4_for_testing

# Create 20 synthetic GW events
create_synthetic_gwtc4_for_testing(
    n_events=20,
    output_file='./test_gwtc4.h5'
)
```

### 3. Train Model (Test Mode)

```bash
python train.py --config configs/default_config.yaml --test-mode
```

This trains for 10 epochs with dummy data (takes ~2 minutes on CPU, faster on GPU)

### 4. Examine Results

```bash
# View training progress
tensorboard --logdir runs/

# Check checkpoints
ls checkpoints/

# View logs
cat logs/training.log
```

## Run Full Pipeline

For production research:

### 1. Generate Full COMPAS Ensemble

```bash
# WARNING: This takes hours to days!
python -m compas_ensemble.generate_ensemble \
    --compas-binary /Users/josephrodriguez/ASTROTHESIS/COMPAS/src/bin/COMPAS \
    --output-dir ./compas_ensemble_output \
    --n-systems 100000 \
    --n-alpha-points 10
```

Expected: ~1000 runs, 100K systems each

### 2. Download Real GWTC-4 Data

Visit: https://zenodo.org/record/8177023/files/GWTC-4_posteriors.h5

Save to: `./gwtc4_data/GWTC-4_posteriors.h5`

### 3. Train Production Model

```bash
python train.py --config configs/default_config.yaml
```

Trains for 100 epochs. Monitor with TensorBoard.

### 4. Test Falsification Criteria

```python
from falsification.test_criteria import FalsificationTester
from data.gwtc4_loader import GWTC4Loader
import torch

# Load model and data
model = torch.load('checkpoints/best_model.pth')
loader = GWTC4Loader('./gwtc4_data/GWTC-4_posteriors.h5')
events = loader.load_all_events()

# Test
tester = FalsificationTester(model, events)
results = tester.run_all_tests()

# Check results
print(f"Criterion 1: {'FALSIFIED' if results['criterion_1']['falsified'] else 'PASSED'}")
print(f"Criterion 2: {'FALSIFIED' if results['criterion_2']['falsified'] else 'PASSED'}")
```

## Common Issues

### COMPAS Not Found
```bash
# Check COMPAS binary exists
ls /Users/josephrodriguez/ASTROTHESIS/COMPAS/src/bin/COMPAS

# If missing, rebuild COMPAS
cd /Users/josephrodriguez/ASTROTHESIS/COMPAS/src
make
```

### CUDA Out of Memory
Reduce batch size in `configs/default_config.yaml`:
```yaml
training:
  batch_size: 128  # or 64
```

### Import Errors
Make sure package is installed:
```bash
pip install -e .
```

## Next Steps

- Read full [README.md](README.md) for detailed documentation
- Explore `notebooks/` for analysis examples
- Modify `configs/default_config.yaml` for your experiments
- Check falsification results in `falsification_results/`

## Key Files

- `compas_ensemble/generate_ensemble.py`: Generate COMPAS simulations
- `models/physics_informed_nn.py`: Neural network architecture
- `train.py`: Main training script
- `falsification/test_criteria.py`: Falsification testing
- `configs/default_config.yaml`: Configuration

## Getting Help

1. Check the README.md
2. Review configuration options in `configs/default_config.yaml`
3. Look at example notebooks
4. Check COMPAS documentation: https://compas.science/

## Citation

If you use this code, please cite:

```bibtex
@software{gw_formation_channels,
  author = {Rodriguez, Joseph},
  title = {Physics-Informed Deep Learning for GW Formation Channel Inference},
  year = {2025},
  institution = {Astronomy Club at UCSD}
}
```

---

**Happy inferring!** 🌌

