# Gravitational Wave Formation Channels Research

**TL;DR:** Multi-code, physics-informed SBI framework for gravitational-wave formation-channel inference. Treats population-synthesis simulators as Bayesian priors to expose epistemic uncertainty. Success/failure is judged by explicit falsification tests where model disagreement dominating observations or failure to isolate α_CE leads to rejection.

This repository contains research code for investigating gravitational-wave formation channels using population-synthesis simulators (COMPAS, COSMIC, POSYDON planned) combined with physics-informed modeling, simulation-based inference (SBI), and domain-adaptation techniques.

**Project Scope:** Undergraduate-led, extensible research framework with intentionally staged components; designed to be foundational, falsifiable, and ready for future extensions without implying missing work.

### Research Question

How can a **physics-informed deep learning architecture** use an ensemble of **population-synthesis codes (COMPAS, COSMIC, POSYDON)** as **Bayesian priors** to jointly perform **simulation-based inference** and **domain adaptation** on **gravitational-wave data**, thereby quantifying both **epistemic uncertainty** (stellar-evolution model disagreement) and **aleatoric uncertainty** (detector noise) in **formation-channel likelihoods**?

**This framework is explicitly falsifiable.**
- **Failure mode: Epistemic dominance.** If ensemble-based **mutual information across code predictions** exceeds **observational uncertainty** for **>50%** of **GWTC-4** events, stellar-evolution systematics dominate and formation-channel inference is unreliable.
- **Failure mode: CE ineffectiveness.** If cross-modal attention fails to isolate **common-envelope efficiency (α_CE)** as the primary driver of Channel I/IV degeneracy (**rank correlation < 0.5**), the CE-governed diversity hypothesis is rejected.

If either criterion fails, the scientific hypothesis is rejected.

## Why This Matters

- GW catalogs now require joint astrophysics + ML pipelines to separate formation-channel populations with quantified uncertainty.
- Multi-code population synthesis exposes model systematics that routinely dominate inference but are rarely treated explicitly.
- This repo provides falsifiable hypotheses, reproducible pipelines, and AWS-ready tooling for community-scale ensemble generation.
- Cross-modal interpretability (attention + causal ranking) ties ML predictions back to the underlying stellar-physics knobs.

## Intended Audience

- Astrophysics researchers working on GWTC-4 formation-channel populations
- ML-for-science practitioners interested in simulation-based inference
- Students and researchers extending population-synthesis pipelines

## Project Overview

This project uses physics-informed modeling with population-synthesis priors and simulation-based inference to:
- Generate synthetic populations of binary compact objects using COMPAS
- Train machine learning models to identify formation-channel populations
- Perform Bayesian inference on gravitational-wave observations
- Test and falsify astrophysical models using GWTC-4 (Gravitational Wave Transient Catalog) data

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

- ✔ COSMIC ensemble generator runs end-to-end locally via `pipelines.ensemble_generation.cosmic.generate_ensemble --test-run`.
- ✔ Multi-code unified generator executes COMPAS + COSMIC test runs via `pipelines.ensemble_generation.multi_code.unified_generator --test-run`.
- ✔ SBI training loop runs with default config (`pipelines.inference_and_falsification.train --config configs/training/pipeline/default_config.yaml`); inference entrypoint available.
- 🟡 COMPAS production path validated on AWS (see `docs/operations/AWS_CLUSTER.md`); local macOS runs are intentionally offloaded.
- 🟡 POSYDON CLI wrapper is wired but awaits grid assets/MESA executables before full activation.
- ⏳ Full epistemic mutual-information study across codes and GWTC-4 event-level exports (figures/tables) planned post-POSYDON grid activation.

## Key Features

- **COMPAS Integration**: Generate realistic populations of compact binary systems
- **Physics-Informed Modeling**: Population-synthesis priors tightly constrain ML components
- **Simulation-Based Inference**: Bayesian parameter estimation using neural density estimators
- **Falsification Framework**: Rigorous statistical testing of astrophysical models
- **GWTC-4 Data Integration**: Analysis pipeline for gravitational wave observations

## Expected Scientific Outcomes

- Channel posteriors with explicit **formation-channel** likelihoods and uncertainty decomposition (epistemic vs aleatoric) per GWTC-4 event.
- Pass/fail verdicts against the two falsification criteria (epistemic dominance, CE ineffectiveness) to assess model viability.
- Constraints on key hyperparameters (e.g., **α_CE**, kicks, winds) reported with mutual-information diagnostics across codes.
- Comparative evidence for where simulator disagreement dominates vs. detector noise, guiding follow-on simulator improvements.

## Architecture & Falsification Plan

The full multi-layer architecture, loss decomposition, and falsification workflow now live in [`docs/overview/ARCHITECTURE.md`](docs/overview/ARCHITECTURE.md). That document covers each layer (population synthesis, observational encoders, cross-modal fusion, SBI heads), the two formal falsification criteria, and implementation status (COMPAS + COSMIC operational, POSYDON planned). Architecture diagram (placeholder) is referenced there.

## Why Multi-Code Matters

Single-code population synthesis can understate model-systematic uncertainty; comparing COMPAS (rapid, high-volume), COSMIC (alternate rapid recipes), and POSYDON (planned detailed MESA grids) surfaces disagreement as epistemic signal. This framework treats simulators as priors, aligns their outputs, and measures mutual information across codes—maintaining physics interpretability rather than acting as a black-box ML pipeline.

Not a black-box: ML components are constrained by physics, and interpretability is a requirement, not an afterthought.

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
- [POSYDON Integration](docs/simulator_notes/POSYDON_INTEGRATION.md) — Environment + roadmap
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
- Cite this repository (pre-publication): “ASTROTHESIS (2025), multi-code GW formation-channel inference framework, UC San Diego.” A DOI will be added upon preprint release.
- Contributions: open issues for questions/feature requests; submit PRs for fixes/experiments following project conventions.

## License

This project is part of research conducted at UC San Diego

## Contact

For questions or collaboration inquiries, please open an issue on this repository.

## Full Pipeline (Conceptual Overview)

```
POPULATION SYNTHESIS (Forward Models)
│
├── COMPAS
│     ├── Rapid binary evolution (analytic prescriptions)
│     ├── Large hyperparameter grid (α_CE, σ_kick, winds, MT efficiency)
│     └── High-volume Monte Carlo populations (~10^6 binaries/model)
│
├── COSMIC
│     ├── Alternate rapid-evolution engine (Hurley-style recipes)
│     ├── Parallel hyperparameter grid overlapping with COMPAS
│     └── Used for model disagreement within rapid codes
│
└── POSYDON (planned)
      ├── MESA-based detailed stellar evolution grids
      ├── Interpolation + ML-flowchart execution
      └── High-fidelity benchmark populations (~10^4-10^5 binaries/model)

----------------------------------------------------------------

SELECTION FUNCTION & OBSERVABLE GENERATION
│
├── Convert source-frame -> detector-frame masses
├── Apply cosmology & metallicity evolution
├── Compute merger redshift distribution
├── Apply LIGO/Virgo selection effects (SNR, detectability)
└── Produce simulated GW observables:
      (m1, m2, χ_eff, distance/redshift, detection probability)

----------------------------------------------------------------

DATA ALIGNMENT (Domain Adaptation Layer)
│
├── Load GWTC-4 real-event posteriors
│     ├── (m1, m2, χ_eff) posterior samples
│     ├── distance/redshift posteriors
│     └── event metadata (detector network, SNR)
│
├── Map real events -> latent observation space
└── Map simulated events -> same latent space
      (removes simulator-detector mismatch)

----------------------------------------------------------------

SIMULATION-BASED INFERENCE (Neural Density Estimation)
│
├── Training Inputs:
│     ├── θ (model hyperparameters)
│     └── simulated GW populations (aligned)
│
├── Neural components:
│     ├── event encoder (per-event summary)
│     ├── population encoder (set-based fusion)
│     ├── cross-modal attention (links events <-> θ)
│     └── density estimator (NPE / NSF / NRE)
│
└── Output:
      Posterior p(θ | GW data)

----------------------------------------------------------------

FORMATION-CHANNEL INFERENCE
│
├── Infer probabilities for channels:
│     ├── isolated binary evolution (IB)
│     ├── common-envelope dominant (CE)
│     ├── chemically homogeneous evolution (CHE)
│     ├── dynamical (GC/NSC)
│     └── other subchannels
│
└── Produce channel-level likelihoods + uncertainties

----------------------------------------------------------------

EPISTEMIC + ALEATORIC UNCERTAINTY DECOMPOSITION
│
├── Epistemic (model disagreement):
│     ├── COMPAS vs COSMIC vs POSYDON distributions
│     ├── mutual information across code predictions
│     └── cross-code variance in θ posterior
│
└── Aleatoric (detector noise):
      ├── GWTC-4 posterior sample width
      └── injection-based calibration (optional)

----------------------------------------------------------------

FALSIFICATION FRAMEWORK
│
├── Test 1: Epistemic Dominance
│     If MI_across_codes > observational_uncertainty
│     for >50% of events -> astrophysical model invalid.
│
└── Test 2: CE Ineffectiveness
      If cross-modal attention fails to assign α_CE
      as main driver of Channel I/IV separation
      (rank correlation < 0.5) -> hypothesis falsified.

----------------------------------------------------------------

RESULTS & EXPORTS
│
├── Figures (mass, spin, redshift distributions)
├── Channel posteriors
├── Hyperparameter constraints
├── Falsification metrics
└── Tables for publication (CSV/LaTeX)
```
