# Quick Reference - GW Formation Channels Environment

## Activate Environment
```bash
conda activate gw_channels
```
or
```bash
source activate_env.sh
```

## Common Commands

### Generate COMPAS Ensemble
```bash
python -m pipelines.ensemble_generation.compas.generate_ensemble \
  --output-dir ./experiments/runs/compas_ensemble_output
```

### Train Model
```bash
python -m pipelines.inference_and_falsification.train \
  --config configs/training/pipeline/default_config.yaml
```

### Run TensorBoard
```bash
tensorboard --logdir results/logs/tensorboard
```

### Run Jupyter
```bash
jupyter notebook  # or jupyter lab
```

### Deactivate Environment
```bash
conda deactivate
```

## Run Without Activating
```bash
/opt/anaconda3/envs/gw_channels/bin/python your_script.py
```

## Check Environment
```bash
conda activate gw_channels
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'MPS: {torch.backends.mps.is_available()}')"
```

## Installed Python Version
- Python 3.10.19

## Key Package Versions
- PyTorch 2.9.1 (with Apple Silicon MPS support)
- NumPy 2.2.6
- All packages from requirements.txt installed

## Environment Location
```
/opt/anaconda3/envs/gw_channels
```

## Troubleshooting

**ModuleNotFoundError?**
→ Make sure environment is activated: `conda activate gw_channels`

**Import errors?**
→ Verify installation: `/opt/anaconda3/envs/gw_channels/bin/python -c "import numpy, torch, yaml"`

**Need to reinstall?**
→ `conda activate gw_channels && pip install -r configs/infrastructure/requirements.txt`

## Project Paths
- **Project Root:** `/Users/josephrodriguez/ASTROTHESIS`
- **COMPAS:** `/Users/josephrodriguez/ASTROTHESIS/simulators/compas`

---
For detailed setup information, see [SETUP.md](SETUP.md)

