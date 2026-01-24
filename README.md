# Gravitational Wave Formation Channels Research

A multi-code pipeline for **formation-channel inference** on **GWTC-4** using **simulation-based inference (SBI)** and **population-synthesis ensembles**.

**What it does**
- Generates BBH populations with **COMPAS** and **COSMIC** (POSYDON planned), applies selection effects, and produces detector-frame observables.
- Trains neural density estimators to infer **channel fractions** and **population hyperparameters** from GWTC-4 posteriors.

**What's different**
- Treats multiple simulators as an **ensemble prior** and turns **cross-code disagreement** into a measurable scientific object (where do models diverge, and why?).

## Scientific Motivation

**The question:** Can we reliably infer which astrophysical formation channels (isolated binary evolution including common-envelope phases, dynamical capture, chemically homogeneous evolution) produced the black hole mergers in GWTC-4?

**The challenge:** Population-synthesis codes disagree on predictions even for the same input physics. If this disagreement exceeds what the data can resolve, channel inference is unreliable.

**Our approach:** Treat code disagreement as structured and measurable. Use an ensemble of simulators (COMPAS, COSMIC, POSYDON) as priors, align their outputs with real data via domain adaptation, and apply simulation-based inference to recover posteriors. Build in diagnostics that flag when systematics dominate and inference becomes unreliable.

**Key contribution:** Cross-code epistemic uncertainty is not treated as noise to average out, but as a diagnostic signal localized in parameter space (e.g., "disagreement spikes at low metallicity + intermediate α_CE"). Models expose where they fail and why.


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

## Features

- **Multi-code ensemble:** COMPAS and COSMIC generate independent populations; POSYDON (planned) provides a detailed-evolution benchmark. Cross-code disagreement is treated as a measurable diagnostic, not averaged away.
- **Selection-function realism:** Cosmology + metallicity evolution, detector-frame conversion, and detectability weights to match catalog-level observables.
- **Domain adaptation:** Latent-space alignment of simulated outputs with GWTC-4 posteriors to reduce simulator-to-reality shift before inference.
- **Simulation-based inference:** Neural density estimation (e.g., normalizing flows) to infer posteriors over population hyperparameters and channel fractions.
- **Falsification + diagnostics:** Operational criteria (defined in `docs/overview/ARCHITECTURE.md`) flag when simulator systematics prevent reliable inference.
- **Interpretability as an output:** Disagreement maps, code-identifiability probes, and attention/feature diagnostics are saved in `results/` as first-class artifacts.


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


## Limitations & Assumptions

- Rapid vs. detailed evolution: COMPAS/COSMIC provide rapid prescriptions; POSYDON detailed grids are pending for high-fidelity validation.
- Grid resolution: Hyperparameter grids are finite; MI diagnostics will highlight where resolution is limiting.
- Selection-function approximations: Current selection uses catalog-level detectability; end-to-end injection campaigns are out of scope for now.
- Cosmology and metallicity evolution follow baseline assumptions; alternative cosmologies can be swapped via configs.
- Detector-noise treatment relies on GWTC-4 posteriors; no reprocessing of raw strain is performed here.


## Full Pipeline

See [`docs/overview/ARCHITECTURE.md`](docs/overview/ARCHITECTURE.md) for the full conceptual and implementation breakdown.
