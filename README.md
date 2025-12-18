# Gravitational Wave Formation Channels Research

**TL;DR:** A multi-code, physics-informed simulation-based inference framework for gravitational-wave formation-channel inference. Population-synthesis simulators (COMPAS, COSMIC, POSYDON planned) are treated as an ensemble of priors to quantify and localize epistemic disagreement. The pipeline aligns simulations with GWTC-4 data, infers channel fractions and hyperparameters, and applies falsification tests when model disagreement dominates observational uncertainty.

This repository contains research code for investigating gravitational-wave formation channels using population-synthesis simulators (COMPAS, COSMIC, POSYDON planned) combined with physics-informed modeling, simulation-based inference (SBI), and domain-adaptation techniques.

**Project Scope:** Extensible research framework with staged components; designed to be falsifiable and ready for future extensions without implying missing work. Focus: elevate epistemic disagreement into a structured, physical object, localized in parameter/metallicity/observable space, and map it to specific simulator assumptions that fail.

### Research Question

How can a **physics-informed deep learning architecture** use an ensemble of **population-synthesis codes (COMPAS, COSMIC, POSYDON)** as **Bayesian priors** to jointly perform **simulation-based inference** and **domain adaptation** on **gravitational-wave data**, and, critically, treat epistemic disagreement as a structured, parameter-localizable, astrophysically informative signal (not just noise) that maps to specific failing assumptions, while also quantifying **aleatoric uncertainty** (detector noise) in **formation-channel likelihoods**?

**This framework is explicitly falsifiable.**
- **Failure mode: Epistemic dominance.** If ensemble-based **mutual information across code predictions** exceeds **observational uncertainty** for **>50%** of **GWTC-4** events, stellar-evolution systematics dominate and formation-channel inference is unreliable.
- **Failure mode: CE ineffectiveness.** If cross-modal attention fails to isolate **common-envelope efficiency (α_CE)** as the primary driver of Channel I/IV degeneracy (**rank correlation < 0.5**), the CE-governed diversity hypothesis is rejected.

If either criterion fails, the scientific hypothesis is rejected.

## Scientific Motivation

- GW catalogs need joint astrophysics + ML to separate formation-channel populations with quantified uncertainty.
- Multi-code population synthesis surfaces simulator systematics that single-code studies miss.
- Domain alignment and explicit falsification tests guard against overconfident inference when simulator disagreement dominates observations.

### Overview & Motivation

Multi-code gravitational-wave formation-channel inference pipeline that combines population-synthesis simulators (COMPAS, COSMIC, POSYDON planned), detector selection modeling, a domain-adaptation layer, and simulation-based inference. The simulators act as an ensemble of Bayesian priors; the neural density estimator learns posteriors over astrophysical hyperparameters and channel fractions using aligned real (GWTC-4) and simulated observations. Realism: rapid and detailed binary evolution, cosmology- and metallicity-aware selection, and neural density estimation already standard in the field. Novelty: explicit multi-code epistemic quantification, domain adaptation to close the simulator-to-reality gap, and built-in falsification tests that reject the model when code disagreement dominates or common-envelope (α_CE) signatures fail to appear.

## Intended Audience

- Astrophysics researchers working on GWTC-4 formation-channel populations
- ML-for-science practitioners interested in simulation-based inference
- Students and researchers extending population-synthesis pipelines

## Project Overview

This project uses physics-informed modeling with population-synthesis priors and simulation-based inference to:
- Generate synthetic populations of binary compact objects using COMPAS, COSMIC, and (planned) POSYDON to expose epistemic spread across codes.
- Apply selection effects, cosmology, and metallicity evolution to produce detector-frame observables comparable to GW catalogs.
- Train domain-adapted neural density estimators for joint population and channel inference on GWTC-4.
- Decompose epistemic vs. aleatoric uncertainty and run explicit falsification tests on astrophysical assumptions.

### Simulator Integration Status

- **COMPAS:** Integrated and validated locally + AWS (Nov 2025)
- **COSMIC:** Integrated for rapid local prototyping (Nov 2025)
- **POSYDON:** Wired into ensemble framework; requires grid setup (Dec 2025)

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

## Current End-to-End Capability

- COSMIC ensemble generator runs end-to-end locally via `pipelines.ensemble_generation.cosmic.generate_ensemble --test-run`.
- Multi-code unified generator executes COMPAS + COSMIC test runs via `pipelines.ensemble_generation.multi_code.unified_generator --test-run`.
- SBI training loop runs with default config (`pipelines.inference_and_falsification.train --config configs/training/pipeline/default_config.yaml`); inference entrypoint available.
- COMPAS production path validated on AWS (see `docs/operations/AWS_CLUSTER.md`); local macOS runs are intentionally offloaded.
- POSYDON CLI wrapper is wired but awaits grid assets/MESA executables before full activation.
- Full epistemic mutual-information study across codes and GWTC-4 event-level exports (figures/tables) planned post-POSYDON grid activation.

## Scientific Positioning

Extended thesis, pillars (disagreement maps, code identifiability, failure taxonomy, causal stress tests, single astrophysical claim), and delivery checklist live in [`docs/overview/SCIENTIFIC_POSITION.md`](docs/overview/SCIENTIFIC_POSITION.md).

## Key Features

- **Multi-code population synthesis**: COMPAS, COSMIC, POSYDON (planned) treated as an ensemble prior to expose code-level epistemic uncertainty.
- **Selection-function realism**: Cosmology + metallicity evolution, detector-frame conversion, and SNR-based detectability weights.
- **Domain adaptation**: Latent-space alignment of simulated and real GW events to reduce simulator-to-reality shift.
- **Simulation-based inference**: Neural density estimators (normalizing flows) with event encoders, population aggregation, and cross-modal attention.
- **Falsification framework**: Automatic rejection when model disagreement dominates observations or when α_CE is not an informative driver; upgraded with ranked failure taxonomy.
- **Disagreement cartography**: Mutual-information maps over θ, metallicity, and chirp mass to localize where simulators diverge.
- **Code identifiability**: p(code | observables) head to expose each simulator’s inductive biases and code-invariant features.
- **Causal stress tests**: Counterfactual runs with shared α_CE/kick prescriptions to separate parameterization vs hidden-physics disagreement.

## Expected Scientific Outcomes

- Channel posteriors with explicit **formation-channel** likelihoods and uncertainty decomposition (epistemic vs aleatoric) per GWTC-4 event.
- Pass/fail verdicts against the two falsification criteria (epistemic dominance, CE ineffectiveness) to assess model viability.
- Constraints on key hyperparameters (e.g., **α_CE**, kicks, winds) reported with mutual-information diagnostics across codes, localized by metallicity/redshift and chirp mass.
- Comparative evidence for where simulator disagreement dominates vs. detector noise, guiding follow-on simulator improvements, including ranked failure modes.
- At least one sharp astrophysical claim enabled by the above (e.g., channel identifiability thresholds in chirp mass or metallicity-conditioned α_CE identifiability).

## Architecture & Falsification Plan

The full multi-layer architecture, loss decomposition, and falsification workflow now live in [`docs/overview/ARCHITECTURE.md`](docs/overview/ARCHITECTURE.md). That document covers each layer (population synthesis, observational encoders, cross-modal fusion, SBI heads), the two formal falsification criteria, and implementation status (COMPAS + COSMIC operational, POSYDON planned). Architecture diagram (placeholder) is referenced there.

## Why Multi-Code Matters

Single-code population synthesis can understate model-systematic uncertainty; comparing COMPAS (rapid, high-volume), COSMIC (alternate rapid recipes), and POSYDON (planned detailed MESA grids) surfaces disagreement as epistemic signal. This framework treats simulators as priors, aligns their outputs, and measures mutual information across codes, maintaining physics interpretability rather than acting as a black-box ML pipeline.

Not a black-box: ML components are constrained by physics, and interpretability is a requirement, not an afterthought.

## Pipeline (high level)

1. Generate populations with COMPAS/COSMIC (POSYDON planned) across α_CE, kicks, winds, etc.
2. Apply cosmology, metallicity evolution, and detectability to produce detector-frame observables.
3. Align simulated and real GW events in a shared latent space to reduce simulator–detector mismatch.
4. Perform simulation-based inference to recover hyperparameters and formation-channel fractions.
5. Quantify epistemic vs. aleatoric uncertainty and apply falsification tests when disagreement dominates observations.

## Outputs

- Posterior samples for key hyperparameters (e.g., α_CE, kick dispersion, winds, metallicity scalings) accounting for multi-code variance.
- Branching fractions and per-event probabilities for formation channels, plus falsification verdicts.
- Predictive distributions (mass, spin, redshift) and mutual-information diagnostics separating epistemic vs. aleatoric contributions.
- Figures/tables for publication: corner plots, channel fraction tables, event-level channel posteriors, and falsification summaries.

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
# POSYDON (grid wrapper)
python -m pipelines.ensemble_generation.posydon.generate_ensemble \
  --posydon-args-file configs/simulator_templates/POSYDON_CLI_ARGS.example \
  --grid-point-count 40 \
  --output-dir ./experiments/runs/posydon_ensemble_output


# For production COMPAS runs, use AWS cluster
# See docs/operations/AWS_CLUSTER.md
```

**Recommendations:**
- **COSMIC**: Use for rapid local prototyping (~seconds per 100 systems)
- **COMPAS**: Deploy on AWS for production runs (hours-days per 10k systems)
- **POSYDON**: Use the CLI wrapper once the grid + MESA executables are configured
- **Multi-code**: Essential for epistemic uncertainty quantification

See [COSMIC Integration Guide](docs/simulator_notes/COSMIC_INTEGRATION.md) for details.

**COMPAS Status:** Local macOS run aborted (Nov 26, 2025), migrating to AWS  
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
