# Scientific Positioning

**Purpose:** State the scientific thesis and epistemic framing.  
**How to use:** Use as the reference for claims/pillars; link to architecture and pipeline docs when implementing or reviewing results.  
**Back to README:** [`README.md`](../../README.md)

Epistemic disagreement is treated as a structured, parameter-localizable, astrophysically informative object. This document captures the scientific thesis, pillars, and falsification stance.

---

## Core Thesis (One Sentence)

“Epistemic disagreement across population-synthesis codes is structured, parameter-localizable, and astrophysically informative — not just noise — and can be mapped onto specific physical assumptions that fail.”

Everything below serves this claim.

---

## Pillar 1 — Turn Code Disagreement Into a Physical Map

**What this does:** Instead of one scalar mutual-information verdict, construct a disagreement phase space:

Disagreement(θ, z, O) = I(code; O | θ, z)

Where:
- θ: hyperparameters (α_CE, kicks, winds…)
- z: redshift / metallicity
- O: observables (masses, χ_eff)

**Why this matters:** Answer “Where in parameter space do simulators disagree — and why?” with statements like:
- “Disagreement spikes at low metallicity and intermediate α_CE”
- “High-mass BBHs are robust across codes; low-mass systems are not”
- “COSMIC vs COMPAS disagree primarily in post-CE mass transfer regimes”

This converts epistemic uncertainty into diagnostic physics.

**Minimal implementation:** Stratified mutual information: bin by α_CE, bin by metallicity, bin by chirp mass.

---

## Pillar 2 — Learn Which Physics Each Code Believes

**What this does:** Train a code-identification head:

p(code | O)

Then ask:
- Which observables allow a network to identify COMPAS vs COSMIC?
- Which features are code-invariant?

**Why this matters:** Learns the inductive bias of each simulator.

Results might look like:
- “Spin–mass correlations uniquely identify COSMIC”
- “Redshift distributions are nearly code-invariant”
- “CE prescriptions leak strongly into χ_eff”

If the code can be identified from observables, epistemic uncertainty is learnable and simulators are making distinct physical claims.

---

## Pillar 3 — Promote Falsification From Binary → Ranked Failure Modes

**What this does:** Replace binary pass/fail with a failure taxonomy:

Failure mode | Physical interpretation  
--- | ---  
High MI at all α_CE | CE modeling fundamentally inconsistent  
MI only at low Z | Stellar winds / metallicity prescriptions diverge  
MI only for high χ_eff | Spin evolution assumptions incompatible  

The pipeline doesn’t just reject models — it states why.

---

## Pillar 4 — Causal Stress Tests

**What this does:** Run counterfactual interventions:

“If all codes share the same α_CE and kick model, does disagreement persist?”

This separates:
- disagreement due to parameterization
- disagreement due to hidden physics choices

---

## Pillar 5 — One Astrophysical Result to Own

One concrete, defensible astrophysical claim enabled by the above. Examples:
- “GWTC-4 BBH channel fractions are identifiable only above X chirp mass”
- “Below Z ≈ 0.2 Z⊙, formation-channel inference is irreducibly epistemic”
- “α_CE is not globally identifiable, but is identifiable in metallicity-conditioned subsets”

---

## Conceptual Stance

From “measuring uncertainty” → “explaining where uncertainty comes from and what it means physically.”