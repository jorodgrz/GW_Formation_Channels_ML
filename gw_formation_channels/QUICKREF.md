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
python -m compas_ensemble.generate_ensemble
```

### Train Model
```bash
python train.py
```

### Run TensorBoard
```bash
tensorboard --logdir runs/
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
→ `conda activate gw_channels && pip install -r requirements.txt`

## Project Paths
- **Project Root:** `/Users/josephrodriguez/ASTROTHESIS/gw_formation_channels`
- **COMPAS:** `/Users/josephrodriguez/ASTROTHESIS/COMPAS`

---
For detailed setup information, see [SETUP.md](SETUP.md)

