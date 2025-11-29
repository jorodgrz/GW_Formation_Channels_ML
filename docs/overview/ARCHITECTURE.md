# Multi-Code Architecture & Falsification Plan

This document details the end-to-end architecture for ASTROTHESIS, spanning population-synthesis ensemble generation, observational encoders, cross-modal alignment, simulation-based inference (SBI), and falsification criteria.

> **Integration status (Nov 2025):** COMPAS ✅, COSMIC ✅, SEVN planned (API assessment complete, generator TBD).

---

## A. Data & Priors Layer (Multi-Code Population Synthesis)

### A1. Ensemble Stellar-Evolution Priors

- Incorporate **COMPAS + COSMIC + SEVN (planned)** as distinct conditional priors:

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

## B. Observational Data Pipeline

### B1. Raw-Strain Encoder

- 1D CNN + dilated WaveNet + transformer encoder to ingest h(t).
- Condition on detector PSD, calibration uncertainty, glitches, and non-Gaussian noise bursts.
- Output latent `z_obs` capturing **aleatoric uncertainty**.

### B2. PE Posterior Encoder (Optional Parallel Path)

- Encode (m₁, m₂, χ_eff, χ_p, z, etc.) via MLP or compact transformer.
- Fuses with raw-strain encoder for hybrid inference, stabilizing early training.

---

## C. Population Synthesis Encoder

### C1. Physics-Aware Embedding Network

- Transformer or graph network mapping simulated population features → latent `z_sim`.
- Conditioning on code `C` ensures the model learns systematic differences.

### C2. Domain-Adaptation Mechanisms

Choose one or combine:

- Adversarial domain adaptation (sim vs real discriminator).
- Normalizing-flow transport from simulation → observational domain.
- Optimal-transport layer to align marginals (Wasserstein).

Goal:

```
z_sim ≈ z_obs  in a common manifold.
```

---

## D. Cross-Modal Alignment Layer

### D1. Cross-Attention Transformer

- Multi-head attention fusing `z_sim ↔ z_obs`.
- Learns causal mappings between physical and observed features.
- Attention maps provide scientific interpretability.

### D2. Causal Interpretability Block

- Extract rank correlations between attention weights and hyperparameters (especially α_CE).
- Enables falsification criterion #2 (α_CE dominance).

---

## E. Simulation-Based Inference Head

### E1. Normalizing-Flow Posterior Estimator

- Replace MDNs with expressive flows (neural spline flows, Fourier flows, diffusion posteriors).
- Outputs:

  ```
  p(θ, C, channel | data)
  ```

### E2. Formation-Channel Classifier

- Softmax head predicting {I, II, III, IV} using fused latent `z_fusion`.

### E3. Uncertainty Decomposition

- **Epistemic:** variance across `p(θ | C, data)` for C ∈ {COMPAS, COSMIC, SEVN}.
- **Aleatoric:** posterior width conditioned on a given code and event.

---

## F. Training Procedure

### F1. Joint Loss

- Likelihood-free objective (NLL on flow outputs).
- Adversarial loss for domain adaptation.
- Calibration loss for uncertainty consistency.
- Cross-modal attention regularization.
- Physics-informed priors (e.g., monotonic trends).

### F2. Two-Stage Training

1. Train domain alignment + encoders (simulated + real).
2. Train the SBI head on the aligned latent space.

---

## G. Hypothesis Testing / Falsification Module

### G1. Falsification Test 1: Epistemic Dominance

Check if:

```
σ_epi > σ_obs
```

for >50 % of GWTC-4 events.  
If true → astrophysical inference is impossible; reject the hypothesis.

### G2. Falsification Test 2: α_CE Causal Dominance

Compute:

```
ρ = rank-correlation(attention weights, α_CE)
```

If `ρ < 0.5`, CE physics does **not** explain channel degeneracy → reject the hypothesis.

---

## Summary

The target system combines multi-code population synthesis (COMPAS, COSMIC, SEVN planned) as hierarchical Bayesian hyperpriors with a joint raw-strain and PE-parameter GW encoder, fused via cross-modal transformers to perform SBI using normalizing flows. Domain adaptation aligns simulated and real distributions through adversarial / OT / flow mapping, while attention-based interpretability extracts causal structure—specifically isolating α_CE as the primary driver of formation-channel diversity. The system quantifies epistemic uncertainty via cross-code disagreement and aleatoric uncertainty via raw-strain noise modeling, and the entire astrophysical hypothesis is rejected if either falsification criterion is triggered.


