# ASTROTHESIS — Gravitational Wave Formation Channels Research

This repository contains research code for investigating gravitational-wave formation channels using population-synthesis simulators (COMPAS, COSMIC, SEVN planned) combined with physics-informed modeling, simulation-based inference (SBI), and domain-adaptation techniques.

### Research Question

How can a physics-informed deep learning architecture use an ensemble of population-synthesis codes (COMPAS, COSMIC, SEVN) as Bayesian priors to jointly perform simulation-based inference and domain adaptation on gravitational-wave data, thereby quantifying both **epistemic uncertainty** (stellar-evolution model disagreement) and **aleatoric uncertainty** (detector noise) in formation-channel likelihoods?

The architecture is considered **falsified** if:

1. **Epistemic dominance:** Ensemble-based epistemic uncertainty (mutual information across code predictions) exceeds observational uncertainty for >50 % of GWTC-4 events, implying stellar-evolution systematics dominate and channels cannot be inferred reliably.
2. **Common-envelope ineffectiveness:** Cross-modal attention fails to isolate common-envelope efficiency (α_CE) as the primary driver of Channel I/IV degeneracy (rank correlation < 0.5), contradicting the working hypothesis that CE physics governs channel diversity.

## Why This Matters

- GW catalogs now require joint astrophysics + ML pipelines to separate formation channels with quantified uncertainty.
- Multi-code population synthesis exposes model systematics that routinely dominate inference but are rarely treated explicitly.
- This repo provides falsifiable hypotheses, reproducible pipelines, and AWS-ready tooling for community-scale ensemble generation.
- Cross-modal interpretability (attention + causal ranking) ties ML predictions back to the underlying stellar-physics knobs.

## Project Overview

This project uses physics-informed modeling with population-synthesis priors and simulation-based inference to:
- Generate synthetic populations of binary compact objects using COMPAS
- Train machine learning models to identify formation channels
- Perform Bayesian inference on gravitational wave observations
- Test and falsify astrophysical models using GWTC (Gravitational Wave Transient Catalog) data

### Simulator Integration Status

- **COMPAS:** Integrated and validated locally + AWS (Nov 2025)
- **COSMIC:** Integrated for rapid local prototyping (Nov 2025)
- **SEVN:** Planned (integration pending)

## Project Structure

```
ASTROTHESIS/
├── README.md                          # Research overview (this file)
├── docs/                              # Living research notes
│   ├── overview/                      # Architecture + big-picture summaries
│   ├── methods/                       # Methodology deep dives
│   ├── operations/                    # Runbooks & environment notes
│   └── simulator_notes/               # Code-specific integration details
├── simulators/                        # External stellar-evolution codes
│   ├── compas/                        # Upstream COMPAS source/build artifacts
│   └── sevn/                          # SEVN source tree and docs
├── pipelines/                         # Python research pipelines
│   ├── ensemble_generation/           # COMPAS/COSMIC/SEVN/multi-code drivers
│   ├── data_alignment/                # GWTC-4 loaders & domain adapters
│   ├── inference_and_falsification/   # Models, SBI, tests, trainers
│   └── shared/                        # Cross-cutting helpers
├── configs/                           # Reusable YAML + infra configs
│   ├── infrastructure/                # Requirements, cluster specs
│   └── training/                      # Experiment configs (default lives here)
├── data/
│   ├── raw/                           # Untouched GW catalogs (ignored by git)
│   └── processed/                     # Feature stores / intermediates
├── experiments/
│   ├── notebooks/                     # Research notebooks (organized by theme)
│   └── runs/                          # Ensemble + training artifacts (with metadata)
├── results/
│   ├── figures/                       # Ready-to-publish visuals
│   ├── tables/                        # CSV/LaTeX tables + falsification exports
│   └── logs/                          # Checkpoints, tensorboard, diagnostics
├── tests/                             # Integration/unit tests
│   └── integration/                   # Minimal ensemble smoke tests
└── scripts/                           # Utility shell helpers (env activation, etc.)
```

## Key Features

- **COMPAS Integration**: Generate realistic populations of compact binary systems
- **Physics-Informed Modeling**: Population-synthesis priors tightly constrain ML components
- **Simulation-Based Inference**: Bayesian parameter estimation using neural density estimators
- **Falsification Framework**: Rigorous statistical testing of astrophysical models
- **GWTC-4 Data Integration**: Analysis pipeline for gravitational wave observations

## Architecture & Falsification Plan

The full multi-layer architecture, loss decomposition, and falsification workflow now live in [`docs/overview/ARCHITECTURE.md`](docs/overview/ARCHITECTURE.md). That document covers each layer (population synthesis, observational encoders, cross-modal fusion, SBI heads), the two formal falsification criteria, and implementation status (COMPAS + COSMIC operational, SEVN planned).

---

## Documentation

- [Quick Reference](docs/operations/QUICKREF.md) — Command palette & daily tasks
- [Setup Guide](docs/operations/SETUP.md) — Environment provisioning and dependencies
- [Quick Start](docs/operations/QUICKSTART.md) — 30-minute onboarding walkthrough
- [Project Summary](docs/overview/PROJECT_SUMMARY.md) — Extended research context
- [Pipeline Overview](docs/overview/PIPELINE_README.md) — Detailed module-by-module tour
- [Architecture & Falsification Plan](docs/overview/ARCHITECTURE.md) — Full multi-code stack and formal tests
- [COMPAS Information](docs/simulator_notes/COMPAS_Info.md) — Simulator build notes
- [COSMIC Integration](docs/simulator_notes/COSMIC_INTEGRATION.md) — Usage guide & status
- [AWS Cluster Playbook](docs/operations/AWS_CLUSTER.md) — Production COMPAS workflow

## Requirements

- Python 3.8+
- COMPAS simulator
- PyTorch
- h5py
- pandas, numpy, scipy
- sbi (simulation-based inference library)

## Installation

See [SETUP.md](docs/operations/SETUP.md) for detailed installation instructions.

```bash
# Create conda environment
conda create -n gw_channels python=3.10
conda activate gw_channels

# Install dependencies from the research-centric layout
pip install -r configs/infrastructure/requirements.txt

# Install the pipelines package
pip install -e .
```

## Usage

### Generate Multi-Code Ensemble

**NEW: COSMIC Integration Complete!** (Nov 26, 2025)

Run ensembles from multiple population synthesis codes for epistemic uncertainty quantification:

```bash
# Quick local test (COSMIC is fast!)
python -m pipelines.ensemble_generation.cosmic.generate_ensemble \
  --test-run --sparse --n-systems 100 \
  --output-dir ./experiments/runs/cosmic_ensemble_output

# Multi-code ensemble (COMPAS + COSMIC)
python -m pipelines.ensemble_generation.multi_code.unified_generator \
  --test-run --sparse --n-systems 100 \
  --codes compas cosmic \
  --output-dir ./experiments/runs/multi_code_ensemble_output

# For production COMPAS runs, use AWS cluster
# See docs/operations/AWS_CLUSTER.md
```

**Recommendations:**
- **COSMIC**: Use for rapid local prototyping (~seconds per 100 systems)
- **COMPAS**: Deploy on AWS for production runs (hours-days per 10k systems)
- **Multi-code**: Essential for epistemic uncertainty quantification

See [COSMIC Integration Guide](docs/simulator_notes/COSMIC_INTEGRATION.md) for details.

**COMPAS Status:** Local macOS run aborted (Nov 26, 2025) - migrating to AWS  
**COSMIC Status:** Fully operational on local macOS (Nov 26, 2025)

### Train Neural Network
```bash
python -m pipelines.inference_and_falsification.train \
  --config configs/training/pipeline/default_config.yaml
```

### Run Inference
```bash
python -m pipelines.inference_and_falsification.inference.sbi_framework
```

## Citation

If you use this code in your research, please cite:
- COMPAS: [Stevenson et al. (2017)](https://arxiv.org/abs/1704.01352)
- This project: [To be published]

## License

This project is part of research conducted at UC San Diego

## Contact

For questions or collaboration inquiries, please open an issue on this repository.

