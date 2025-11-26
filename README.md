# ASTROTHESIS - Gravitational Wave Formation Channels Research

This repository contains research code for investigating gravitational wave formation channels using COMPAS (Compact Object Mergers: Population Astrophysics and Statistics) simulations combined with machine learning techniques.

**Research Question** Research Question: How can a physics-informed deep learning architecture use an ensemble of population synthesis codes (COMPAS, COSMIC, SEVN) as Bayesian priors to jointly perform simulation-based inference and domain adaptation on gravitational wave data, thereby quantifying both astrophysical model uncertainty (epistemic) and detector noise uncertainty (aleatoric) in formation-channel likelihoods? Specifically, the architecture will be falsified if: (1) ensemble epistemic uncertainty (mutual information across code predictions) exceeds observational uncertainty for >50% of GWTC-4 events, indicating that stellar evolution model systematics prevent meaningful channel inference regardless of domain adaptation; or (2) cross-modal attention weights fail to isolate common envelope efficiency (α_CE) as the primary driver of Channel I/IV degeneracy (rank correlation <0.5), contradicting the hypothesis that CE physics dominates formation-channel 

## Project Overview

This project uses physics-informed neural networks and simulation-based inference to:
- Generate synthetic populations of binary compact objects using COMPAS
- Train machine learning models to identify formation channels
- Perform Bayesian inference on gravitational wave observations
- Test and falsify astrophysical models using GWTC (Gravitational Wave Transient Catalog) data

## Project Structure

```
ASTROTHESIS/
├── COMPAS/                    # COMPAS stellar evolution simulator
├── gw_formation_channels/     # Main project code
│   ├── compas_ensemble/       # COMPAS simulation generation
│   ├── data/                  # Data loaders (GWTC-4)
│   ├── models/                # Neural network architectures
│   ├── inference/             # SBI framework
│   ├── falsification/         # Model testing criteria
│   ├── configs/               # Configuration files
│   └── notebooks/             # Jupyter notebooks for analysis
├── my_results/                # Simulation outputs (ignored by git)
└── README.md                  # This file
```

## Key Features

- **COMPAS Integration**: Generate realistic populations of compact binary systems
- **Physics-Informed Neural Networks**: ML models constrained by physical laws
- **Simulation-Based Inference**: Bayesian parameter estimation using neural density estimators
- **Falsification Framework**: Rigorous statistical testing of astrophysical models
- **GWTC-4 Data Integration**: Analysis pipeline for gravitational wave observations

---

# Target Architecture for the Multi-Code Physics-Informed Deep Learning System

## A. Data & Priors Layer (Multi-Code Population Synthesis)

### A1. Ensemble Stellar-Evolution Priors

* Incorporate **COMPAS + COSMIC + SEVN** as *distinct conditional priors*

  ```
  p(θ, C) = p(C) p(θ|C)
  ```

* Harmonize hyperparameter domains (α_CE, σ_kick, β_MT, Z distribution).

* Tag each sample with **code identity embeddings** for uncertainty decomposition.

* Maintain consistent channel labels across codes (I/II/III/IV).

### A2. Simulation–Observation Alignment

* Include selection effects: detector sensitivity, detection probability p_det(θ).

* Generate *detector-frame* distributions: (m1, m2, χ_eff, χ_p, z, e_10Hz).

---

## B. Observational Data Pipeline

### B1. Raw-Strain Encoder

* Use **1D CNN + dilated WaveNet + transformer encoder** to ingest h(t).

* Condition on:

  * Detector PSD

  * Calibration uncertainty

  * Glitches & non-Gaussian noise bursts

* Output latent z_obs capturing **aleatoric uncertainty**.

### B2. PE Posterior Encoder (Optional Parallel Path)

* Encode (m1, m2, χ_eff, χ_p, z, etc.) via MLP or small transformer.

* Fuses with raw-strain encoder for hybrid inference.

* Acts as a stabilizing inductive bias early in training.

---

## C. Population Synthesis Encoder

### C1. Physics-Aware Embedding Network

* Transformer or graph network mapping simulated population features → latent z_sim.

* Conditioning on code C ensures model learns systematic differences.

### C2. Domain Adaptation Mechanisms

Choose one or hybrid:

* **Adversarial domain adaptation** (sim vs real discriminator).

* **Normalizing flow transport** from simulation → observational domain.

* **Optimal transport layer** to align marginals (Wasserstein).

Goal:

```
z_sim ≈ z_obs  in a common manifold.
```

---

## D. Cross-Modal Alignment Layer

### D1. Cross-Attention Transformer

* Multi-head attention fusing z_sim ↔ z_obs.

* Learns causal mappings between physical and observed features.

* Attention maps used to verify astrophysical hypotheses.

### D2. Causal Interpretability Block

* Extract rank correlations between attention weights and hyperparameters (especially α_CE).

* Enables **falsification criterion #2** (α_CE dominance).

---

## E. Simulation-Based Inference Head

### E1. Normalizing Flow Posterior Estimator

* Replace MDN with an expressive flow:

  * Neural spline flows

  * Fourier flows

  * Diffusion posteriors (conditional score models)

* Outputs:

  ```
  p(θ, C, channel | data)
  ```

### E2. Formation-Channel Classifier

* Softmax head predicting {I, II, III, IV}.

* Uses fused latent z_fusion.

### E3. Uncertainty Decomposition

* **Epistemic:** variance across p(θ | C, data) for C ∈ {COMPAS, COSMIC, SEVN}.

* **Aleatoric:** width of posterior conditioned on a given C and event.

---

## F. Training Procedure

### F1. Joint Loss

* Likelihood-free objective (NLL on flow outputs)

* Adversarial loss (for domain adaptation)

* Calibration loss (for uncertainty consistency)

* Cross-modal attention regularization

* Physics-informed inductive priors (e.g., monotonic trends)

### F2. Two-Stage Training

1. **Stage 1:** Train domain alignment + encoders (sim + real).

2. **Stage 2:** Train SBI head on aligned latent space.

---

## G. Hypothesis Testing / Falsification Module

### G1. Falsification Test 1: Epistemic Dominance

Check if:

```
σ_epi > σ_obs
```

for >50% of GWTC-4 events.

If yes → astrophysical inference is impossible; reject hypothesis.

### G2. Falsification Test 2: α_CE Causal Dominance

Compute:

```
ρ = rank-correlation(attention weights, α_CE)
```

If:

```
ρ < 0.5
```

→ CE physics does *not* explain channel degeneracy → reject hypothesis.

---

## Full Architecture Summary

The target system combines multi-code population synthesis (COMPAS, COSMIC, SEVN) as hierarchical Bayesian hyperpriors with a joint raw-strain and PE-parameter GW encoder, fused via cross-modal transformers to perform simulation-based inference using normalizing flows. Domain adaptation aligns simulated and real distributions through adversarial/OT/flow mapping, while attention-based interpretability extracts causal structure—specifically isolating α_CE as the primary driver of formation-channel diversity. The system quantifies epistemic uncertainty via cross-code disagreement and aleatoric uncertainty via raw-strain noise modeling, and the entire astrophysical hypothesis is rejected if code discrepancies dominate the observational uncertainty or if α_CE is not the dominant causal factor in channel-level degeneracies.

---

## Documentation

- [Quick Reference](gw_formation_channels/QUICKREF.md) - Command reference and common operations
- [Setup Guide](gw_formation_channels/SETUP.md) - Environment setup and installation
- [Quick Start](gw_formation_channels/QUICKSTART.md) - Get started quickly
- [Project Summary](gw_formation_channels/PROJECT_SUMMARY.md) - Detailed project overview
- [COMPAS Information](COMPAS_Info.md) - COMPAS simulator details

## Requirements

- Python 3.8+
- COMPAS simulator
- PyTorch
- h5py
- pandas, numpy, scipy
- sbi (simulation-based inference library)

## Installation

See [SETUP.md](gw_formation_channels/SETUP.md) for detailed installation instructions.

```bash
# Create conda environment
conda create -n gw_channels python=3.10
conda activate gw_channels

# Install dependencies
cd gw_formation_channels
pip install -r requirements.txt
```

## Usage

### Generate COMPAS Ensemble (AWS cluster recommended)

- Use a **quick local smoke test** before shipping jobs to AWS:
  ```bash
  cd gw_formation_channels
  python compas_ensemble/generate_ensemble.py --test-run --sparse --n-systems 100
  ```
- Run the **production ensemble** on an AWS cluster following `gw_formation_channels/AWS_CLUSTER.md`.
- Slice the grid on each worker with the new `--start-index` / `--end-index` flags to parallelize runs safely.

**Current Status:** Local macOS run aborted on Nov 26, 2025 after ~36% of the first combination.  
We are migrating the full ensemble build to AWS; progress notes live in `gw_formation_channels/ENSEMBLE_STATUS.md`.

### Train Neural Network
```bash
cd gw_formation_channels
python train.py --config configs/default_config.yaml
```

### Run Inference
```bash
cd gw_formation_channels
python -m inference.sbi_framework
```

## Citation

If you use this code in your research, please cite:
- COMPAS: [Stevenson et al. (2017)](https://arxiv.org/abs/1704.01352)
- This project: [To be published]

## License

This project is part of research conducted at UC San Diego

## Contact

For questions or collaboration inquiries, please open an issue on this repository.

