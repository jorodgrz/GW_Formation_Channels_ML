# Gravitational Wave Formation Channels Research

A multi-code pipeline for **formation-channel inference** on **GWTC-4** using **simulation-based inference (SBI)** and **population-synthesis ensembles**.

**What it does**
- Generates BBH/BNS populations with **COMPAS** and **COSMIC** (POSYDON planned), applies selection effects, and produces detector-frame observables.
- Trains neural density estimators to infer **channel fractions** and **population hyperparameters** from GWTC-4 posteriors.

**What's different**
- Treats multiple simulators as an **ensemble prior** and turns **cross-code disagreement** into a measurable scientific object (where do models diverge, and why?).

**How it stays honest**
- Includes **operational falsification tests** that flag when inference is unreliable due to simulator systematics (see `docs/overview/ARCHITECTURE.md`).

## Quick start

```bash
# 1. Set up environment
conda create -n gw_channels python=3.10 -y
conda activate gw_channels
pip install -r configs/infrastructure/requirements.txt && pip install -e .

# 2. COSMIC quick local test
python -m pipelines.ensemble_generation.cosmic.generate_ensemble \
  --test-run --sparse --n-systems 100 \
  --output-dir ./experiments/runs/cosmic_test

# 3. Train (default config)
python -m pipelines.inference_and_falsification.train \
  --config configs/training/pipeline/default_config.yaml
```

## What you get if you run this

- **Training artifacts:** `results/logs/` (configs, checkpoints, TensorBoard logs)
- **Ensemble outputs:** `experiments/runs/<run_id>/` (HDF5 + metadata)
- **(Full runs) Event-level posteriors:** `results/tables/event_level_posteriors.csv` (channel probabilities per GWTC-4 event)
- **(Full runs) Disagreement maps:** `results/figures/disagreement_map_*.png` (where simulators diverge)
- **(Full runs) Hyperparameter constraints:** Corner plots, posterior samples for α_CE, kicks, winds, metallicity scaling
- **(WIP) Falsification summary:** `results/tables/falsification_summary.csv` (pass/fail diagnostics + MI metrics)

## Scientific Motivation

**The question:** Can we reliably infer which astrophysical formation channels (isolated binary evolution including common-envelope phases, dynamical capture, chemically homogeneous evolution) produced the black hole mergers in GWTC-4?

**The challenge:** Population-synthesis codes disagree on predictions even for the same input physics. If this disagreement exceeds what the data can resolve, channel inference is unreliable.

**Our approach:** Treat code disagreement as structured and measurable. Use an ensemble of simulators (COMPAS, COSMIC, POSYDON) as priors, align their outputs with real data via domain adaptation, and apply simulation-based inference to recover posteriors. Build in diagnostics that flag when systematics dominate and inference becomes unreliable.

**Key contribution:** Cross-code epistemic uncertainty is not treated as noise to average out, but as a diagnostic signal localized in parameter space (e.g., "disagreement spikes at low metallicity + intermediate α_CE"). Models expose where they fail and why.

## Intended Audience

- Astrophysics researchers working on GWTC-4 formation-channel populations
- ML-for-science practitioners interested in simulation-based inference
- Students and researchers extending population-synthesis pipelines

## Status

| Component | Status | Notes |
|-----------|--------|-------|
| **COSMIC ensemble** | ✅ Operational | Fast local prototyping (~seconds per 100 systems) |
| **COMPAS ensemble** | 🟡 Partial | AWS validated; production grid pending |
| **POSYDON** | ⚪ Planned | CLI wrapper ready; awaiting grid download |
| **Multi-code runner** | ✅ Operational | COMPAS + COSMIC test runs functional |
| **SBI training loop** | ✅ Operational | Runs with default config; ready for production |
| **GWTC-4 data** | ✅ Operational | Posteriors loaded; domain adaptation wired |
| **Falsification tests** | 🟡 Partial | Framework designed; awaiting full ensemble runs |

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
│   ├── cosmic/                        # COSMIC configs + project-specific notes
│   └── posydon/                       # POSYDON source tree and docs (planned)
├── pipelines/                         # Python research pipelines
│   ├── ensemble_generation/           # COMPAS/COSMIC/POSYDON/multi-code drivers
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

## Scientific Positioning

For detailed thesis, pillars (disagreement maps, code identifiability, failure taxonomy, causal stress tests), and falsification criteria, see [`docs/overview/SCIENTIFIC_POSITION.md`](docs/overview/SCIENTIFIC_POSITION.md) and [`docs/overview/ARCHITECTURE.md`](docs/overview/ARCHITECTURE.md).

## Key Features

- **Multi-code ensemble:** COMPAS and COSMIC generate independent populations; POSYDON (planned) provides a detailed-evolution benchmark. Cross-code disagreement is treated as a measurable diagnostic, not averaged away.
- **Selection-function realism:** Cosmology + metallicity evolution, detector-frame conversion, and detectability weights to match catalog-level observables.
- **Domain adaptation:** Latent-space alignment of simulated and GWTC-4 posteriors to reduce simulator-to-reality shift before inference.
- **Simulation-based inference:** Neural density estimation (e.g., normalizing flows) to infer posteriors over hyperparameters and channel fractions.
- **Falsification + diagnostics:** Operational criteria (defined in `docs/overview/ARCHITECTURE.md`) to flag when simulator systematics prevent reliable inference.
- **Interpretability as an output:** Disagreement maps, code-identifiability probes, and attention/feature diagnostics are saved in `results/` as first-class artifacts.

## Why Multi-Code

Single-code studies can miss model systematics. Comparing COMPAS (rapid, high-volume) and COSMIC (alternate recipes) exposes where predictions diverge; POSYDON (detailed MESA grids, planned) provides a high-fidelity benchmark. Models expose code-conditional features and disagreement maps; interpretability is built in, not bolted on.

## Pipeline (high level)

1. Generate populations with COMPAS/COSMIC (POSYDON planned) across α_CE, kicks, winds, etc.
2. Apply cosmology, metallicity evolution, and detectability to produce detector-frame observables.
3. Align simulated and real GW events in a shared latent space to reduce simulator–detector mismatch.
4. Perform simulation-based inference to recover hyperparameters and formation-channel fractions.
5. Quantify epistemic vs. aleatoric uncertainty and apply falsification tests when disagreement dominates observations.

---

## Documentation

- [Quick Reference](docs/operations/QUICKREF.md), Command palette & daily tasks
- [Setup Guide](docs/operations/SETUP.md), Environment provisioning and dependencies
- [Quick Start](docs/operations/QUICKSTART.md), 30-minute onboarding walkthrough
- [Project Summary](docs/overview/PROJECT_SUMMARY.md), Current status and implementation overview
- [Scientific Positioning](docs/overview/SCIENTIFIC_POSITION.md), Thesis, pillars, and epistemic framing
- [Pipeline Overview](docs/overview/PIPELINE_README.md), Comprehensive pipeline explanation with scientific context
- [Architecture & Falsification Plan](docs/overview/ARCHITECTURE.md), Full multi-code stack and formal tests
- [COMPAS Information](docs/simulator_notes/COMPAS_Info.md), Simulator build notes
- [COSMIC Integration](docs/simulator_notes/COSMIC_INTEGRATION.md), Usage guide & status
- [POSYDON Integration](docs/simulator_notes/POSYDON_INTEGRATION.md), Environment + roadmap
- [AWS Cluster Playbook](docs/operations/AWS_CLUSTER.md), Production COMPAS workflow

## Requirements

- Python 3.10+ (3.8 may work but 3.10 recommended)
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
# POSYDON (grid wrapper)
python -m pipelines.ensemble_generation.posydon.generate_ensemble \
  --posydon-args-file configs/simulator_templates/POSYDON_CLI_ARGS.example \
  --grid-point-count 40 \
  --output-dir ./experiments/runs/posydon_ensemble_output


# For production COMPAS runs, use AWS cluster
# See docs/operations/AWS_CLUSTER.md
```

**Recommendations:**
- **COSMIC:** Fast local prototyping (~seconds per 100 systems)
- **COMPAS:** Production runs on AWS (hours-days per 10k systems)
- **POSYDON:** CLI wrapper ready; awaiting grid download
- **Multi-code:** Essential for epistemic uncertainty quantification

See [COSMIC Integration Guide](docs/simulator_notes/COSMIC_INTEGRATION.md) and [AWS Cluster](docs/operations/AWS_CLUSTER.md) for details.

### Train Neural Network
```bash
python -m pipelines.inference_and_falsification.train \
  --config configs/training/pipeline/default_config.yaml
```

### Run Inference
```bash
python -m pipelines.inference_and_falsification.inference.sbi_framework
```

## Reproducibility Guarantees

- YAML-driven configs: training settings in `configs/training`, infrastructure specs in `configs/infrastructure`.
- Metadata-logged runs: artifacts, configs, and logs stored in `experiments/runs` and `results/logs`.
- Controlled seeds: default seeds set in configs to enable repeatable runs; override via CLI flags when needed.
- Outputs: figures in `results/figures`, tables in `results/tables`, falsification exports alongside run metadata.

## Limitations & Assumptions

- Rapid vs. detailed evolution: COMPAS/COSMIC provide rapid prescriptions; POSYDON detailed grids are pending for high-fidelity validation.
- Grid resolution: Hyperparameter grids are finite; MI diagnostics will highlight where resolution is limiting.
- Selection-function approximations: Current selection uses catalog-level detectability; end-to-end injection campaigns are out of scope for now.
- Cosmology and metallicity evolution follow baseline assumptions; alternative cosmologies can be swapped via configs.
- Detector-noise treatment relies on GWTC-4 posteriors; no reprocessing of raw strain is performed here.

## Roadmap (Next 3–6 Months)

- Activate POSYDON grid wrapper with validated MESA executables and grid assets.
- Run full epistemic mutual-information study across COMPAS/COSMIC/POSYDON ensembles.
- Publish GWTC-4 event-level channel posteriors with falsification verdicts and exportable tables.
- Release lightweight reproducibility template (config + run metadata snapshot) for external comparisons.
- This framework is intended to support a future preprint when results mature.

## How to Cite / Collaborate

- Cite COMPAS: [Stevenson et al. (2017)](https://arxiv.org/abs/1704.01352).

## License

This project is part of research conducted at UC San Diego

## Contact

For questions or collaboration inquiries, please open an issue on this repository.

## Full Pipeline

See [`docs/overview/ARCHITECTURE.md`](docs/overview/ARCHITECTURE.md) for the full conceptual and implementation breakdown.
