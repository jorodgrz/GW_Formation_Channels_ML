# Project Implementation Summary

**Purpose:** Current status snapshot and implementation overview—what's built, what's in progress, and where components live.  
**How to use:** Review this for project status before diving into code; see `PIPELINE_README.md` for scientific context and `README.md` for quick start.  
**Back to README:** [`README.md`](../../README.md)

---

## Executive Summary

Multi-code, physics-informed simulation-based inference framework for gravitational-wave formation-channel inference. Population-synthesis simulators (COMPAS, COSMIC, POSYDON planned) are treated as ensemble priors to quantify epistemic uncertainty. The pipeline performs domain adaptation to align simulations with GWTC-4 observations, applies neural density estimation for Bayesian inference, and includes explicit falsification tests that reject the model when simulator disagreement dominates or when expected physical drivers fail to emerge.

**Institution:** UC San Diego - ASTROTHESIS  
**Status:** Active Development (December 2025)  
**Stage:** Multi-code ensemble generation operational; SBI training loop functional; end-to-end inference and falsification pending full production runs.

---

## Current Implementation Status

### Completed Components

#### 1. COSMIC Ensemble Generation
- **Location:** `pipelines/ensemble_generation/cosmic/`
- **Status:** Fully operational for rapid local prototyping
- **Capability:** End-to-end sparse grid runs (100-10k systems per parameter combination)
- **Performance:** ~seconds per 100 systems on local macOS
- **Output:** HDF5 files with metadata, parameter grids (α_CE, kicks, winds, metallicity)
- **Integration:** Test runs validated via `--test-run` flag

#### 2. Multi-Code Unified Generator
- **Location:** `pipelines/ensemble_generation/multi_code/`
- **Status:** COMPAS + COSMIC test runs operational
- **Capability:** Runs ensemble across multiple codes with harmonized parameter grids
- **Output:** Unified metadata schema across codes for downstream inference
- **Usage:** `unified_generator --test-run --codes compas cosmic`

#### 3. SBI Training Loop
- **Location:** `pipelines/inference_and_falsification/train.py`
- **Status:** End-to-end training functional with default config
- **Architecture:** Event encoder, population aggregator, cross-modal attention, normalizing flow density estimator
- **Features:** 
  - Multi-objective loss (posterior likelihood, domain adaptation, uncertainty calibration)
  - Checkpointing and TensorBoard logging
  - Gradient clipping and learning rate scheduling
  - YAML-driven configuration (`configs/training/pipeline/default_config.yaml`)
- **Validation:** Runs to completion with synthetic data; ready for production ensembles

#### 4. GWTC-4 Data Infrastructure
- **Location:** `pipelines/data_alignment/` and `data/raw/`
- **Status:** GWTC-4 posterior loader wired and tested
- **Capability:** Ingests event posteriors (m1, m2, χ_eff, distance), computes derived parameters
- **Integration:** Dataset ready for domain adaptation and inference
- **Synthetic Mode:** Can generate synthetic events for testing and validation

#### 5. Configuration & Documentation System
- **Configs:** YAML-driven (`configs/training/`, `configs/infrastructure/`)
- **Documentation:**
  - `README.md` — Entry point with TL;DR, usage, falsification criteria
  - `docs/overview/PIPELINE_README.md` — Comprehensive scientific pipeline explanation
  - `docs/overview/ARCHITECTURE.md` — System architecture and falsification plan
  - `docs/overview/SCIENTIFIC_POSITION.md` — Core thesis and pillars
  - `docs/operations/SETUP.md`, `QUICKSTART.md`, `QUICKREF.md` — Setup and daily operations
  - `docs/simulator_notes/` — Code-specific integration details (COMPAS, COSMIC, POSYDON)
- **Reproducibility:** Controlled seeds, metadata logging, artifact tracking in `experiments/runs/`

### In Progress

#### 1. COMPAS Production Ensemble on AWS
- **Status:** AWS Batch/EC2 validated for COMPAS; production 40-combination sparse grid pending
- **Timeline:** Deploy full grid (~10^6 binaries total), upload to S3, sync locally
- **Blocker:** Awaiting cluster allocation and final grid parameter selection
- **Documentation:** See `docs/operations/AWS_CLUSTER.md` for runbooks

#### 2. Unified Ensemble Loader for Training
- **Task:** Implement `UnifiedEnsembleGenerator.load_ensemble_for_training`
- **Requirement:** Transform COMPAS/COSMIC HDF5 outputs → training tensors (pop_synth_inputs)
- **Integration Point:** Feed into `train.py` and `falsification/` modules
- **Status:** API designed; implementation in progress

#### 3. End-to-End Training Run
- **Task:** Execute full training with production COMPAS + COSMIC ensembles
- **Outputs:** Trained posterior estimator, checkpoints, TensorBoard logs
- **Next Step:** Run `FalsificationTester` to populate `results/tables/falsification`
- **Validation:** Compare posteriors to toy model results; coverage tests

#### 4. Cross-Code Mutual-Information Study
- **Task:** Compute MI diagnostics across COMPAS/COSMIC (and POSYDON when ready)
- **Purpose:** Quantify epistemic uncertainty; feed into Test 1 (epistemic dominance)
- **Outputs:** MI values per event, population-level MI, figures for publication
- **Status:** Framework designed; awaiting production ensembles

#### 5. GWTC-4 Event-Level Exports
- **Task:** Generate per-event channel posteriors, falsification verdicts, exportable tables
- **Format:** CSV/LaTeX tables, corner plots, attention visualizations
- **Location:** `results/tables/`, `results/figures/`
- **Status:** Pending full inference run

### Planned / Pending

#### 1. POSYDON Integration
- **Status:** CLI wrapper implemented; awaits grid assets and MESA executables
- **Next Steps:**
  - Download interpolation grids via `posydon-setup-pipeline --download-grids` (~25 GB)
  - Run 3×100-system smoke test to verify parameter mapping and HDF5 outputs
  - Extend multi-code generator to include POSYDON
  - Update AWS runbooks with POSYDON timing benchmarks
- **Timeline:** Targeted for Q1 2026 pending grid availability

#### 2. Unit & Integration Tests
- **Scope:** Data loaders, ensemble runners, model forward pass, falsification metrics
- **Location:** `tests/integration/`
- **Status:** Smoke tests exist; comprehensive suite pending

#### 3. Publication Outputs
- **Task:** Prepare figures, tables, and text for preprint
- **Dependencies:** Full production runs, falsification results, cross-code MI study
- **Timeline:** Post-milestones above

#### 4. Reproducibility Template
- **Task:** Release lightweight config snapshot for external comparisons
- **Format:** YAML configs + run metadata + instructions
- **Purpose:** Enable community replication and extension

---

## Component Architecture

### Population Synthesis Ensemble

The pipeline uses three population-synthesis codes as ensemble priors:

#### COMPAS (Validated)
- **Type:** Rapid binary evolution (analytic prescriptions)
- **Scale:** ~10^6 binaries per model
- **Grid:** α_CE, natal kicks (σ_kick), winds (β_wind), mass transfer efficiency
- **Deployment:** AWS Batch/EC2 for production; local macOS runs offloaded
- **Status:** Integration complete; production grid pending

#### COSMIC (Operational)
- **Type:** Rapid evolution (Hurley-style recipes, formerly BSE)
- **Scale:** ~10^6 binaries per model
- **Grid:** Parallel to COMPAS (α_CE, kicks, winds) for epistemic comparison
- **Deployment:** Local macOS for rapid prototyping (~seconds per 100 systems)
- **Status:** Fully operational; sparse grids generated

#### POSYDON (Planned)
- **Type:** Detailed MESA-based stellar evolution grids
- **Scale:** ~10^4–10^5 binaries per model (higher fidelity, lower volume)
- **Grid:** Interpolated from detailed pre-computed tracks
- **Deployment:** CLI wrapper ready; awaits grid download
- **Status:** API integrated; activation pending grid assets

### Selection Function & Observables

- **Source → Detector Frame:** Redshift masses, luminosity distance under ΛCDM cosmology
- **Cosmology & Metallicity:** Star formation history, Z evolution (e.g., Z(z) scaling)
- **Selection Effects:** LIGO/Virgo sensitivity (SNR > threshold), detection probability weights
- **Outputs:** Synthetic GW catalogs (m1, m2, χ_eff, z, p_det) per simulation

### Data Alignment (Domain Adaptation)

- **Real Data:** GWTC-4 posterior samples ingested from `data/raw/`
- **Latent Embedding:** Encoder maps events → shared latent space (simulated + real)
- **Adaptation Techniques:** 
  - Adversarial discriminator (real vs. sim)
  - Maximum Mean Discrepancy (MMD) loss
  - Optional optimal transport alignment
- **Goal:** Remove simulator-detector mismatch; align distributions in latent space

### Simulation-Based Inference (Neural Density Estimation)

- **Architecture:**
  - Event encoder (per-event summary)
  - Population aggregator (set-based attention or pooling)
  - Cross-modal attention (events ↔ hyperparameters θ)
  - Normalizing flow density estimator (Neural Posterior Estimation)
- **Training:**
  - Simulated experiments: sample θ → generate population → label with θ
  - Loss: maximize p(θ | data) under flow output
  - Amortized: train once, infer instantly
- **Outputs:** Posterior p(θ | GWTC-4) for hyperparameters + channel fractions

### Formation-Channel Inference

- **Channels:**
  - Isolated binary evolution (IB)
  - Common-envelope dominant (CE)
  - Chemically homogeneous evolution (CHE)
  - Dynamical (globular/nuclear clusters, GC/NSC)
  - Other subchannels (triples, primordial, etc.)
- **Inference Modes:**
  - Population-level: branching fractions per channel (e.g., 60% IB, 30% dynamical, 10% CHE)
  - Per-event: probability distribution over channels for each GW event
- **Interpretability:** Cross-modal attention highlights which events inform which channels

### Epistemic & Aleatoric Uncertainty Decomposition

#### Epistemic (Model Uncertainty)
- **Cross-code disagreement:** MI between code predictions and observables
- **Posterior variance:** Width in θ posterior due to simulator differences
- **Quantification:** Stratified MI, cross-code θ variance

#### Aleatoric (Data Uncertainty)
- **Event posteriors:** Width in GWTC-4 parameter estimates (detector noise)
- **Sample variance:** Finite N events → statistical fluctuations
- **Quantification:** Posterior sample spread, 1/√N scaling checks

### Falsification Framework

#### Test 1: Epistemic Dominance
- **Criterion:** If MI(code; observables) > σ_obs for >50% of events → model invalid
- **Interpretation:** Simulator disagreement exceeds data information content
- **Action:** Reject inference; improve models or gather more data

#### Test 2: CE Ineffectiveness
- **Criterion:** If rank_correlation(α_CE, channel-separating latent variables) < 0.5 → hypothesis falsified
- **Interpretation:** Common-envelope efficiency does not emerge as key driver
- **Action:** Reject CE-dominance hypothesis; explore alternative physics

### Outputs & Results

- **Posteriors:** θ (α_CE, kicks, winds, etc.) with credible intervals
- **Channel Fractions:** Branching ratios + per-event probabilities
- **Figures:**
  - Mass/spin/redshift distributions (model vs. data)
  - Corner plots (θ joint/marginal posteriors)
  - Attention heatmaps (event × parameter importance)
- **Tables:**
  - Hyperparameter constraints (median + 90% CI)
  - Channel fractions (% per channel with uncertainties)
  - Per-event classifications (event ID, channel probabilities)
  - Falsification verdicts (pass/fail + MI/correlation values)
- **Exports:** CSV/LaTeX for publication

---

## File Structure & Key Modules

```
ASTROTHESIS/
├── pipelines/
│   ├── ensemble_generation/
│   │   ├── compas/                  # COMPAS ensemble generator
│   │   ├── cosmic/                  # COSMIC ensemble generator (operational)
│   │   ├── posydon/                 # POSYDON CLI wrapper (pending)
│   │   └── multi_code/              # Unified multi-code runner
│   ├── data_alignment/              # GWTC-4 loaders, domain adaptation
│   ├── inference_and_falsification/
│   │   ├── models/                  # Neural architectures (encoders, flows)
│   │   ├── inference/               # SBI framework (NPE)
│   │   ├── falsification/           # Test criteria (MI, α_CE correlation)
│   │   └── train.py                 # End-to-end training loop
│   └── shared/                      # Cross-cutting utilities
├── configs/
│   ├── training/pipeline/           # Training configs (default_config.yaml)
│   └── infrastructure/              # Requirements, cluster specs
├── data/
│   ├── raw/                         # GWTC-4 posteriors (gitignored)
│   └── processed/                   # Feature stores, intermediates
├── experiments/
│   ├── notebooks/                   # Research notebooks
│   └── runs/                        # Ensemble + training artifacts
├── results/
│   ├── figures/                     # Publication-ready plots
│   ├── tables/                      # CSV/LaTeX exports
│   └── logs/                        # Checkpoints, TensorBoard
├── docs/
│   ├── overview/                    # PIPELINE_README, ARCHITECTURE, SCIENTIFIC_POSITION
│   ├── operations/                  # SETUP, QUICKSTART, AWS_CLUSTER
│   └── simulator_notes/             # COMPAS_Info, COSMIC_INTEGRATION, POSYDON_INTEGRATION
└── tests/
    └── integration/                 # Smoke tests, integration checks
```

---

## Usage Quick Reference

### Generate Ensembles

```bash
# COSMIC (fast local prototyping)
python -m pipelines.ensemble_generation.cosmic.generate_ensemble \
  --test-run --sparse --n-systems 100 \
  --output-dir ./experiments/runs/cosmic_ensemble_output

# Multi-code (COMPAS + COSMIC)
python -m pipelines.ensemble_generation.multi_code.unified_generator \
  --test-run --sparse --n-systems 100 \
  --codes compas cosmic \
  --output-dir ./experiments/runs/multi_code_ensemble_output

# POSYDON (pending grid activation)
python -m pipelines.ensemble_generation.posydon.generate_ensemble \
  --posydon-args-file configs/simulator_templates/POSYDON_CLI_ARGS.example \
  --grid-point-count 40 \
  --output-dir ./experiments/runs/posydon_ensemble_output
```

### Train Model

```bash
python -m pipelines.inference_and_falsification.train \
  --config configs/training/pipeline/default_config.yaml
```

### Run Inference

```bash
python -m pipelines.inference_and_falsification.inference.sbi_framework
```

---

## Next Milestones

1. **Deploy COMPAS Production Grid on AWS** — Run 40-combination sparse grid, sync to local
2. **Implement Unified Ensemble Loader** — Transform HDF5 → training tensors
3. **Execute End-to-End Training** — Full production run with COMPAS + COSMIC ensembles
4. **Run Falsification Tests** — Populate `results/tables/falsification`
5. **Cross-Code MI Study** — Quantify epistemic uncertainty across codes
6. **Generate GWTC-4 Event-Level Outputs** — Channel posteriors, figures, tables
7. **Activate POSYDON** — Download grids, validate, integrate into multi-code pipeline
8. **Prepare Publication Outputs** — Draft figures/tables for preprint

---

## Scientific Context

For comprehensive explanation of the pipeline's scientific motivation, technical implementation, and why each stage is both realistic and novel, see:

- **[Pipeline Overview](PIPELINE_README.md)** — Full narrative with realism vs. novelty breakdown
- **[Architecture](ARCHITECTURE.md)** — System design and falsification criteria
- **[Scientific Position](SCIENTIFIC_POSITION.md)** — Core thesis and pillars

For operational details:

- **[README](../../README.md)** — Entry point with quick start
- **[Setup Guide](../operations/SETUP.md)** — Environment provisioning
- **[AWS Cluster](../operations/AWS_CLUSTER.md)** — Production COMPAS workflow

---

## Contributing & Reproducibility

- **Controlled Seeds:** Default seeds in configs; override via CLI for variations
- **Metadata Logging:** All runs tracked in `experiments/runs/` with configs + timestamps
- **YAML-Driven:** Training settings, infrastructure specs fully specified
- **Open Issues:** For questions, feature requests, or collaboration inquiries
- **External Replication:** Reproducibility template planned post-publication

---

## License & Citation

**Institution:** UC San Diego  
**Project:** ASTROTHESIS

**Cite COMPAS:** Stevenson et al. (2017) — https://arxiv.org/abs/1704.01352  
**Cite This Repo (pre-publication):** "ASTROTHESIS (2025), multi-code GW formation-channel inference framework, UC San Diego." DOI pending preprint release.
