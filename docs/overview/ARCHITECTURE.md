# Multi-Code Architecture & Falsification Plan

**Purpose:** Describe the end-to-end architecture: population synthesis, encoders, alignment, SBI, and falsification criteria.  
**How to use:** Read top-down when wiring or auditing components; pair with `README.md` for orientation and `PIPELINE_README.md` for module-level detail.  
**Back to README:** [`README.md`](../../README.md)

> **Integration status (Nov 2025):** COMPAS operational, COSMIC operational, POSYDON planned (API assessment complete, generator TBD).

---

## Pipeline Summary (Conceptual)

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

---

## A. Data & Priors Layer (Multi-Code Population Synthesis)

### A1. Ensemble Stellar-Evolution Priors

- Incorporate **COMPAS + COSMIC + POSYDON (planned)** as distinct conditional priors:

  ```
  p(θ, C) = p(C) p(θ | C)
  ```

- Harmonize hyperparameter domains (α_CE, σ_kick, β_MT, Z distribution).
- Tag each sample with **code identity embeddings** for downstream uncertainty decomposition.
- Maintain consistent channel labels across codes (I/II/III/IV).

### A2. Simulation–Observation Alignment

- Include selection effects: detector sensitivity, detection probability p_det(θ).
- Generate detector-frame distributions: (m₁, m₂, χ_eff, χ_p, z, e_10Hz).

---

## B. Data Alignment (Domain Adaptation)

**What lives here:** Map GWTC-4 posterior samples and simulated detected events into a shared latent space; optional adversarial/OT alignment to reduce simulator–detector mismatch.

**Outputs:** Aligned latents for real and simulated events.

---

## C. Inference & Fusion

**What lives here:** Event encoder, population/set encoder, cross-modal attention linking events ↔ θ, and a neural density estimator (NPE/NSF/NRE) to produce p(θ | data).

**Outputs:** Posteriors over θ given aligned data.

---

## D. Formation-Channel Inference

**What lives here:** Channel head producing probabilities for IB, CE, CHE, dynamical (GC/NSC), and other subchannels; channel-level likelihoods and uncertainties.

---

## E. Uncertainty Decomposition

**What lives here:** Epistemic from cross-code disagreement and θ-posterior spread; aleatoric from GW posterior widths and finite sample size; mutual-information diagnostics optional.

---

## F. Falsification Criteria

- **Epistemic dominance:** If MI_across_codes > observational_uncertainty for >50% of events, the model is invalid.  
- **CE ineffectiveness:** If cross-modal importance fails to surface α_CE as a driver (rank correlation < 0.5), the hypothesis is rejected.

---

## G. Outputs

- Posteriors for θ and channel probabilities, MI diagnostics, falsification metrics, and publication-ready figures/tables.

---

## H. Training Procedure (high level)

- Train alignment + encoders on simulated + real data.
- Train the SBI head on the aligned latent space.
- Monitor MI across codes and α_CE salience for falsification readiness.


