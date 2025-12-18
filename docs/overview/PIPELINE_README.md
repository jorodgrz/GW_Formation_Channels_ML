# Multi-Code Gravitational-Wave Formation Channel Inference Pipeline

**Purpose:** Comprehensive explanation of the pipeline: scientific motivation, technical implementation, and context for each stage.  
**How to use:** Read this for an in-depth understanding of how the pipeline works, why it's designed this way, and what makes it both realistic and novel.  
**Back to README:** [`README.md`](../../README.md)

---

## Overview and Motivation

Gravitational-wave observations of merging compact binaries carry signatures of how those systems formed, but interpreting these signals requires linking them to complex astrophysical models. The pipeline outlined here aims to do exactly that: it combines multiple population-synthesis simulations with a physics-informed deep learning framework to infer the formation channels of binary black hole mergers from data. In essence, it treats simulations (from codes like COMPAS, COSMIC, and POSYDON) as an ensemble of Bayesian priors and uses simulation-based inference (SBI) to compare those predictions with real gravitational-wave events. This approach is grounded in current astrophysical practice – and pushes into new territory:

- **Realism:** Each component of the pipeline corresponds to state-of-the-art methods or tools already used in the field (e.g. rapid binary evolution codes, LIGO/Virgo selection models, neural density estimators). Previous studies have shown the need to consider multiple formation scenarios and varied physics to explain the diversity of LIGO/Virgo events, and that hundreds of GW detections can indeed constrain binary evolution parameters like common-envelope efficiency or supernova kicks. Our pipeline builds directly on such insights.

- **Novelty:** Unlike conventional analyses, this framework explicitly integrates multiple simulation codes for epistemic uncertainty quantification, employs a domain adaptation layer to align simulations with real data, and incorporates falsification tests that will reject the model if it fails key consistency checks. This goes beyond earlier workflows that either used a single population model or lacked a built-in way to gauge when model systematics dominate.

Below, we break down each stage of the pipeline, explaining its function and how it contributes to a realistic yet cutting-edge inference on GW formation channels.

---

## Population Synthesis (Forward Models)

The pipeline begins by generating synthetic populations of binary compact objects under various astrophysical assumptions. This is done with an ensemble of population-synthesis codes, each representing a different approach to modeling binary star evolution:

### COMPAS

A rapid binary population synthesis code (Compact Object Mergers: Population Astrophysics and Statistics) that uses analytic prescriptions to evolve binaries efficiently. COMPAS can simulate on the order of 10^6 binary systems per model, enabling broad exploration of uncertain parameters like the common-envelope efficiency (α_CE), supernova natal kick velocities, wind mass-loss rates, and more. For example, Barrett et al. (2018) varied four such COMPAS parameters (including α_CE and kick velocity dispersion) to assess how well future GW data could constrain them. This demonstrates COMPAS's ability to rapidly produce large populations across a range of hyperparameter settings.

### COSMIC

An alternative rapid population synthesis code built on the Hurley et al. stellar evolution formulas (formerly known as BSE). COSMIC is a community-developed tool (Breivik et al. 2020) designed for quick simulations of binary populations. It provides a valuable independent check: by using a different code with its own implementations of binary physics, we can see how sensitive our results are to modeling choices. Both COMPAS and COSMIC can generate millions of binaries in days of compute time, but their small differences (e.g. in how they treat stellar winds or supernova kicks) will later serve as signals of model uncertainty. In the pipeline, we run COSMIC on a parallel grid of hyperparameters similar to COMPAS's (e.g. varying α_CE, natal kick dispersion, mass transfer efficiency) to directly compare outcomes.

### POSYDON (Planned Integration)

A next-generation binary population synthesis framework that uses detailed stellar evolution tracks from the MESA code. POSYDON forgoes some speed to gain fidelity: it computes extensive grids of binary evolution models with full stellar structure, covering many initial conditions. These grids (on the order of 10^4–10^5 binaries per set) are then interpolated to synthesize populations. The inclusion of POSYDON (once its grid is incorporated) means our pipeline can benchmark the rapid codes against high-fidelity predictions. For instance, where COMPAS/COSMIC rely on simplified recipes, POSYDON follows a binary through mass transfer phases, common-envelope events, and supernovae with a detailed physics engine. This helps identify if the rapid codes are missing important effects. POSYDON is also able to simulate binaries across a range of metallicities and cosmic time, incorporating star formation history in the universe, which is crucial for comparing to observed gravitational-wave sources (which come from various redshifts and hence different stellar populations).

### Multi-Code Strategy

Using three codes in concert is a deliberate strategy. By treating each simulator's output as a sample from the "prior" (our astrophysical beliefs) and later comparing them, the pipeline can capture theoretical uncertainties. If all codes agree, we gain confidence that certain features (say, a predicted black hole mass distribution shape) are robust. If they disagree, those differences signal epistemic uncertainty in our modeling. This multi-code approach is relatively novel; previous analyses have typically used one code or a few fixed models. In one comparable study, Zevin et al. (2021) analyzed GW data with a suite of five different binary evolution models (covering both isolated and dynamical formation scenarios) and found that considering multiple models was essential – no single channel could explain more than ~70% of the observations. Our pipeline builds this multi-model ethos in from the start, before fitting the data, by generating an ensemble of populations under varied assumptions.

---

## Selection Function & Synthetic Observables

Simulating astrophysical binaries is only the first step – we must next determine which of those simulated systems would actually be observable by gravitational-wave detectors like LIGO/Virgo (and at what redshifts, with what measured properties). This stage applies the selection function and generates observables in the detector frame:

### Source-Frame to Detector-Frame Conversion

The masses and other parameters from population synthesis are initially in the source frame (e.g. the binary's rest frame). We convert them to the redshifted detector frame. For a given simulated binary merging at redshift z, the pipeline redshifts the masses (m_det = (1+z) * m_source) and accounts for the luminosity distance corresponding to that z under a chosen cosmology. We also assign each simulated merger a random sky location and orientation, and we know which detector network (e.g. LIGO Hanford+Livingston+Virgo) we're considering for selection.

### Cosmology & Metallicity Evolution

Because the universe's properties change with redshift, we incorporate models for how star formation rate and metallicity vary over cosmic time. The pipeline weights the simulated binaries according to a cosmic star-formation history and metallicity-specific formation rate. (In practice, this can mean sampling more low-metallicity binaries at high redshift, etc., reflecting that early universe stars had lower metal content.) As an example, the POSYDON framework explicitly accounts for cosmological metallicity evolution by providing model grids at different Z and weighting them appropriately. We leverage such approaches so that the synthetic populations are realistic for the universe's timeline – e.g., more high-Z binaries form later in cosmic history. This step yields a merger redshift distribution for each model, i.e. how many mergers we expect at each z.

### Detector Selection Effects

Real GW detectors have limited reach – they preferentially catch the loudest signals. The pipeline applies selection criteria mimicking the LIGO/Virgo search sensitivity. Typically, a network signal-to-noise ratio (SNR) threshold (around 8–12) is used as a proxy for "detectability." We compute each binary's expected SNR (given its distance, masses, etc.) and accept it as "detected" if SNR exceeds the threshold. Alternatively, we assign a detection probability to each event based on its parameters. This captures Malmquist bias, where heavier, nearer, or higher-spin-aligned binaries are over-represented in the detected sample. Accounting for this bias is crucial – otherwise comparisons between models and data can be skewed. In standard population analyses, this is done by estimating the fraction of simulated mergers that would be observable by the detectors. We do the same: each simulated merger gets a weight or a binary flag (detected/not) based on the selection function.

### Simulated GW Observables

After applying detection cuts, we compile the properties of the "detected" subset of each simulation. For each such event we record the primary mass (m1), secondary mass (m2), effective spin (χ_eff), and the redshift (or luminosity distance) at detection. These are precisely the quantities reported for real GW events in catalogs like GWTC-4. By also noting the detection probability or network SNR, we have all the ingredients to compare with the real catalog. Essentially, this stage outputs a synthetic gravitational-wave catalog for each population-synthesis model.

By the end of this step, the pipeline has moved from theoretical predictions (all mergers that happen in the universe) to observable predictions (mergers that LIGO/Virgo would see). This ensures that when we later align simulations with actual data, we're comparing apples to apples.

---

## Data Alignment (Domain Adaptation Layer)

A major challenge in simulation-based inference is the simulator-to-reality gap: even after accounting for selection biases, there can be discrepancies between simulated data and real data due to simplifications in simulations or noise/artifacts in real measurements. The pipeline addresses this via a domain adaptation step that brings simulated and actual observations into a common "latent" representation space.

### Real GW Events (GWTC-4) as Data

We ingest the real catalog data, specifically the posterior samples for each event's parameters. For each detected GW event, LIGO/Virgo provides posterior distributions for quantities like m1, m2, χ_eff, and luminosity distance. Rather than just a single point estimate, we use the full posterior samples to represent the uncertainty in each event. We also take event metadata (e.g. which detectors observed it, network SNR, etc.) if needed, since that could affect the domain mapping (for example, events with 3-detector observations might have better constrained parameters than 2-detector events).

### Latent Space Embedding

We train or define an encoder that maps each event (or each set of posterior samples for an event) to a point in a latent feature space. Think of this like a transformation of the raw input (masses, spins, distances) into a new set of features that capture the essential information in a more abstract way. Crucially, we do this for both real events and simulated events. For real events, the encoder might take into account the spread of the posterior (reflecting uncertainty due to detector noise). For simulated events, which are "noise-free" truth values with an assigned detection probability, we might add some equivalent scatter or simply encode them with deterministic features. The goal is to have a representation where the distributions of latent features for real vs. simulated data can be aligned.

### Domain Adaptation Technique

To align these distributions, the pipeline can employ methods from transfer learning. One approach is to include a domain-adversarial training objective: an auxiliary network tries to distinguish whether a latent vector came from a simulation or from real data, and the encoder learns to fool this discriminator, thereby making the latent representations indistinguishable between domains. Another approach (used in similar SBI contexts) is to add a loss term like Maximum Mean Discrepancy (MMD) between the latent feature distribution of simulated and real datasets. In essence, we penalize the encoder if it produces latent embeddings that have different statistical properties for simulations vs. real events. Prior research on simulation-based inference has noted that without such adaptation, a network trained purely on simulations might face domain shift when applied to real data. Our pipeline preempts this by explicitly training the encoder to remove detectable differences between how simulated and actual GW events are represented in latent space.

### Aligned "Observation" Space

After this step, both the real GWTC-4 events and the simulated detected events are mapped into the same latent observation space. Ideally, a given latent vector could equally likely correspond to a real event or a simulation. This means the subsequent inference network doesn't have to worry about, say, systematic offsets in parameter distributions or the presence of noise only in real data – those have been normalized out. It "levels the playing field." In similar contexts (e.g., strong gravitational lensing), adding a domain adaptation loss made an SBI model more robust when transitioning from noiseless simulations to real noisy observations. In our case, we ensure that differences like detector noise, parameter estimation uncertainty, or any simplified physics in simulations do not cause the inference to misinterpret real data. The mapping essentially absorbs those differences as nuisance factors.

By the end of the data alignment stage, we have two sets of comparable high-level data: one from the real world (with dozens of GW events embedded as latent vectors, each capturing their mass/spin/etc information with uncertainty) and one from each simulation model (with thousands of synthetic events embedded similarly). Both live in the same vector space. This setup is now ready for the core inference engine to evaluate which astrophysical model (or which parameters) best explain the real observations.

---

## Simulation-Based Inference (Neural Density Estimation)

At the heart of the pipeline is a neural inference model that learns the relationship between the astrophysical model parameters and the observed data. We utilize modern simulation-based inference techniques, specifically neural density estimation, to extract posterior distributions of model parameters θ given the data. Here's how it works and why it's powerful:

### Inference Goal

We want p(θ | D_GW) – the posterior probability of the hyperparameters θ (which include things like α_CE, supernova kick velocity dispersion, metallicity-related factors, and possibly channel fractions) given the observed GW data D_GW (the GWTC-4 events). Directly writing down a likelihood for this is intractable because we don't have a simple analytical model for how θ produces the data; we only have our simulations. Traditional hierarchical Bayesian analyses tackle this with Monte Carlo sampling and importance weighting, which can be extremely slow for complex models and large event catalogs. Instead, we adopt amortized inference: using simulations to train a neural network that directly approximates the posterior.

### Training Data for SBI

We generate many simulated "experiments" by sampling different θ values (from some prior range) and, for each, producing a simulated population of GW events (through the population synthesis + selection steps described above). Each such simulation (after alignment) yields a set of latent observation vectors. We label this set with the θ that generated it. These paired examples – (simulated events latent vectors, θ) – form the training data for the neural network. Essentially, we're teaching the network: "if you see data that looks like this, it likely came from astrophysical parameters like that." Over many examples, the network learns to invert the simulator.

### Neural Architecture

We use a set-based neural network architecture, reflecting that each "dataset" (population of GW events) is a collection of varying size, and order doesn't matter. A convenient design is to have:

1. **Event encoder** that takes an individual event's latent features (from the alignment stage) and produces an embedding or summary for that event
2. **Population aggregator** that combines the embeddings of all events in a set (simple pooling like averaging or something more expressive like a transformer-based encoder that uses attention across events)
3. **Cross-modal attention mechanism** that connects the event-level representation with the parameter representation. For instance, we might have the network try to attend to which events are most informative about each component of θ. This is inspired by architectures in multi-modal learning. Here one "modal" input is the set of events and the other is a candidate θ (or some encoding of θ). This cross-attention helps the model learn, for example, which observed mergers are telling it about the common-envelope efficiency vs. which are telling it about, say, the presence of dynamical formation.

### Density Estimator

The network includes a density estimation head that outputs an approximation to p(θ | data). There are a few approaches: Neural Posterior Estimation (NPE) trains a network to output the parameters of the posterior directly (treating it as, say, a mixture of Gaussians or other distribution). Neural Likelihood Estimation (NLE) or Neural Ratio Estimation (NRE) instead learn the likelihood or likelihood ratio, but ultimately yield the same ability to sample the posterior. In our pipeline, we lean on normalizing flows – powerful neural networks capable of modeling complex probability distributions. A normalizing flow can serve as the conditional posterior model, taking as input the summary of the data and producing a sample or density for θ. Prior works have shown normalizing flows are effective for GW population inference.

### Training Procedure

The neural network is trained by maximizing the likelihood of the true θ under the network's predicted posterior for each simulated dataset (or minimizing a divergence measure between the predicted and true distributions). Thanks to amortization, this training is expensive upfront but, once trained, inference is virtually instantaneous – the network can produce p(θ|D_GW) without further simulations. This is a huge advantage given the complexity of our model. As a sanity check, we validate the network on simulations (e.g., using coverage tests or comparing with brute-force likelihood-free inference on toy problems). The SBI approach has been found to yield accurate posteriors in astrophysical cases while using the full information in the data, rather than relying on a few summary statistics. For instance, it doesn't compress the data down to, say, just the event rate and mean mass – it considers potentially all details (mass distribution shape, spin correlations, etc.) that the network can learn to exploit.

### Physics-Informed Design

A noteworthy aspect is that our architecture isn't a black box doing arbitrary pattern recognition – it's structured to reflect the problem. By having an event encoder and a population-level module, we acknowledge the two-level hierarchy (many events per population). By including cross-attention between events and parameters, we imbue a form of interpretability: the network can, in principle, indicate which events are most informative about which parameters. This addresses a common critique of ML models in science – we want to trace predictions back to physical causes. In our results, we could examine the attention weights to see, for example, if high-mass-ratio events drive the inference of certain supernova kick values, or if a particular event with an outlier spin is influencing the χ_eff distribution inference, etc.

The outcome of this stage is a trained neural posterior estimator ready to take the actual observed GW data (in latent space) and compute the posterior distribution of astrophysical hyperparameters p(θ | GW data). This posterior encapsulates what we can learn about binary evolution physics from the current GW catalog. Notably, this approach allows simultaneous inference of multiple parameters and even mixture fractions for different channels in one coherent framework – something very hard to do with traditional methods due to the high dimensionality and simulator complexity.

---

## Formation-Channel Inference

One of the primary goals of this project is to determine the astrophysical formation channels of the observed gravitational-wave events. In practice, this means assessing the probability that a given merger (or a fraction of mergers) came from different evolutionary pathways such as:

- **Isolated binary evolution (IB):** Classic field binaries evolving in isolation, possibly involving stable mass transfer or common-envelope phases.
- **Common-envelope dominant (CE):** A subset of isolated evolution where a common-envelope phase is the crucial step in bringing the binary close enough (this might be considered part of isolated binaries, but the pipeline separates it as a channel to specifically evaluate the impact of the CE process).
- **Chemically homogeneous evolution (CHE):** A channel where two massive stars in a binary mix their envelopes (due to rapid rotation and tidal locking), evolving nearly chemically homogeneously and thus avoiding expansion; they can collapse into a close black-hole binary without a traditional CE phase.
- **Dynamical formation (GC/NSC):** Channels where binaries form or harden via dynamical interactions in dense stellar environments – e.g., in globular clusters (GC) or nuclear star clusters (NSC). These tend to involve capture or exchange encounters and can produce different signatures (like more isotropic spin orientations, higher eccentricities at formation, etc.).
- **Other subchannels:** This could include things like triples leading to mergers via Kozai-Lidov resonance, primordial black hole binaries (if considered), or any scenario outside the main four above.

### Learning Channel Fractions

The neural network, by training on simulations that include mixtures of channels, learns to identify signatures of each channel in the data. For example, dynamical formation in clusters might produce more mergers with misaligned spins or certain mass ratio distributions. CHE might produce characteristically higher aligned spins but only at certain mass ranges (since very massive, low-metallicity binaries are needed). By presenting these patterns in the training data, the network can disentangle them in the observed catalog. Previous population analyses using Bayesian mixture models found that multiple channels are indeed needed to explain LIGO/Virgo data. Zevin et al. (2021) found that a mix of isolated and dynamical models was strongly favored over any single channel dominating, with each channel contributing at most ~70% of events. Our approach aims to infer exactly those kinds of mix ratios, but with greater detail and in a single coherent inference rather than a separate step-wise fit.

### Per-Event Classification vs Global Fractions

The pipeline can also infer on a per-event basis which channel is most likely. This is a more fine-grained analysis: given an event's parameters (masses, spin, etc.), what's the probability it originated from, say, a dynamical encounter in a globular cluster versus from isolated binary evolution? This essentially is a classification problem for each event, informed by the population-level context. Early attempts at this have been made (e.g., to classify individual black hole mergers by origin) using cuts or simpler machine learning on posterior samples. Our pipeline's advantage is that it does this within the full Bayesian model – so it uses all event information and the knowledge of the overall population simultaneously. In practice, the cross-modal attention in our network can facilitate this: it can learn to assign an event embedding high attention with a certain channel label. The result would be a probability distribution for each event across channels, which we can then marginalize to get the overall fractions.

### Interpretability and Consistency

Because channel identification is tied to model parameters in our approach, it provides physical insight. For instance, if the posterior shows a high probability that a given event is from the dynamical channel, that is likely because the event has characteristics that the isolated models (with any α_CE) struggle to produce, but a dynamical model matches. This could be a heavy total mass with low spins, for example, which isolated models find hard to explain if pair-instability mass gaps or spin alignments are at play, whereas dynamical formation might more easily produce such an event via random pairing of black holes and isotropic spins. By training on simulations, these subtle differences can be learned by the network.

In summary, the formation-channel inference layer translates the abstract outputs of the neural network into astrophysically meaningful quantities: how many mergers come from each pathway, and which pathway likely produced each merger. This is the ultimate connection back to the big question of the project – understanding the diversity of cosmic origins for the newfound population of binary black holes.

---

## Epistemic and Aleatoric Uncertainty Decomposition

A standout feature of this pipeline is its ability to disentangle epistemic uncertainty (uncertainty in the models/parameters) from aleatoric uncertainty (uncertainty in the data/measurements). In complex inference problems, it's immensely helpful to know whether improving the data (e.g., with more observations or better detectors) versus improving the models (e.g., better physics or additional codes) would reduce uncertainty. We achieve this decomposition as follows:

### Epistemic Uncertainty (Model Uncertainty)

This arises because our astrophysical models are an approximation of reality – different codes or different choices of parameters might fit the data similarly well. In the pipeline, we quantify epistemic uncertainty in two ways:

1. **Ensemble model disagreement:** Since we have multiple simulation codes, we can examine the variance in their predictions. Concretely, for a given set of hyperparameters θ, COMPAS might predict one distribution of observables and COSMIC another. We can compute statistical measures like the KL divergence or mutual information between the predictions of different codes. If the codes strongly disagree, the mutual information between "which code was used" and the outcome will be high. The pipeline specifically calculates the mutual information across code predictions for each event or for the population as a whole. If this is significant, it flags that the results depend on which theoretical model is chosen – a hallmark of epistemic uncertainty. As a rule of thumb, we compare this to the observational uncertainty. If model-to-model differences are larger than the spread allowed by data noise, that's a warning sign.

2. **Posterior sensitivity to hyperparameters:** Even within one code's results, epistemic uncertainty appears as a broad posterior over the hyperparameters θ. A narrow posterior means the data pinpoint a parameter value; a wide posterior means many values could explain the data. Some of that width, however, might be because different assumptions (codes) would shift the best-fit value. We can gauge this by running the inference separately for each simulator and comparing the posteriors for θ. For instance, perhaps using COMPAS alone we get α_CE posterior peaked at 2–3, but with COSMIC alone it peaks at 4–5. The cross-code variance in the inferred θ is another measure of epistemic uncertainty. Our final combined posterior (marginalizing over codes) will be broader if the codes disagree significantly. By attributing portions of that broadness to code differences, we effectively label it epistemic.

### Aleatoric Uncertainty (Data/Statistical Uncertainty)

This stems from the fact that we have a finite number of noisy observations. In GW astronomy, each event's parameters aren't exact – they have a posterior distribution due to instrument noise and degeneracies, and we only have on the order of O(100) events so far, which is a limited sample. We capture aleatoric uncertainty through:

1. **Event posterior sampling:** The pipeline uses the full posterior samples of each GW event. If an event's parameters are poorly constrained (wide posterior), that uncertainty propagates into the population inference. For example, a poorly measured spin on one event will just contribute a broad likelihood over χ_eff to the population model. Conversely, if an event is very well measured (narrow posterior), it provides a sharper piece of evidence. By incorporating the entire posterior, we don't treat the data as "certain" points; the spread in those samples is a direct measure of observational noise effects (aleatoric uncertainty). Technically, this enters our inference as a kind of likelihood width: broader event posteriors lead to broader population likelihood.

2. **Statistical sample variance:** We have only N events – if N were infinite, we'd nail down the population; at N=70 (say), there's statistical fluctuation. We often quantify this via credible intervals on inferred rates or distribution parameters. In our neural approach, this will appear as the posterior width even if models were fixed. If we somehow had no model uncertainty (say all codes agreed perfectly and the model was basically correct), the remaining width in p(θ|D) would essentially be due to the finite dataset and noise – i.e. aleatoric. We can increase the number of simulated observations (via importance weighting or hypothetical future detections) to see how the posterior shrinks – if it shrinks with 1/√N behavior, that was aleatoric-dominated. If it doesn't shrink much even with many events, that suggests model (epistemic) uncertainties dominate.

By quantifying both types, the pipeline can decompose the total uncertainty in any inference result. For example, suppose we infer α_CE to be between 2 and 5 (a wide range). We could report that epistemic uncertainty (differences between COMPAS and COSMIC and the lack of POSYDON data in that range) contribute, say, 70% of that range, whereas aleatoric uncertainty (the limited number of detected mergers and their measurement errors) contribute 30%. Such a statement is extremely informative: it tells theorists whether collecting more data would tighten the constraint or if, instead, they need better physics in the models.

---

## Falsification Framework

Science advances not just by confirming hypotheses, but also by falsifying them when they don't hold up. A core principle of this project is that it's explicitly falsifiable: we've built in criteria to decide if our astrophysical models (and the assumptions therein) are inconsistent with the gravitational-wave data. Rather than quietly getting a poor fit and tweaking the model ad hoc, we have clear tests that will wave a red flag. Two key falsification tests are implemented:

### Test 1: Epistemic Dominance (Model Invalidity)

After performing inference, we examine the relative size of epistemic vs. aleatoric uncertainties for the events. If we find that for >50% of the events (or for the population overall) the model disagreements exceed the observational uncertainties, then our astrophysical modeling is deemed inadequate. In other words, if the differences between COMPAS, COSMIC, and POSYDON predictions are larger than the differences between the model predictions and the actual data, we have a serious problem: we could fit the data equally well with very different physics assumptions. Formally, we might check if the mutual information between the simulation code and the latent space predictions is high, or if the posterior over θ is multi-modal corresponding to different codes. If so, the pipeline will conclude that stellar-evolution systematics dominate the inference, rendering formation channel conclusions unreliable. This is essentially a falsification of the ensemble model – it tells us that current population synthesis models are too inconsistent with each other (or too flexible) to pinpoint how these black holes formed. The recommended action would be to improve the physics in the models or bring in additional observational constraints (for instance, electromagnetic observations of progenitors) before retrying the inference. It's a fail-safe against overconfident claims: rather than reporting a spurious tight constraint that is artifact of assuming one particular code, we'd report "model not validated – differences between models exceed the information content of the data."

### Test 2: CE Ineffectiveness (Hypothesis Rejection)

One of the central hypotheses in isolated binary evolution is that the common-envelope efficiency (α_CE) is a critical parameter that influences outcomes and helps distinguish different evolutionary channels (like standard isolated vs. alternative pathways). Our pipeline uses a cross-modal attention mechanism partly to probe this: we expect that if α_CE truly governs, say, whether a binary ends up in Channel I (isolated classical) or Channel IV (say, dynamical or some distinct channel), the network should learn to focus on α_CE when trying to differentiate those cases. We will evaluate something like the rank correlation between α_CE and the network's latent features or attention scores that separate channels. If the pipeline fails to highlight α_CE – for example, if the attention weights or learned importance for α_CE are low, or if varying α_CE doesn't significantly change the model's ability to fit different subsets of events – then that suggests the data do not support the notion that α_CE is the dominant factor in channel differentiation. In simpler terms, this would falsify the hypothesis that "common-envelope efficiency is the main driver of the differences between at least two formation channels." The threshold mentioned (rank correlation < 0.5 between α_CE and channel-separating latent variables) is an arbitrary but concrete criterion: below that, we say the network isn't picking up a strong α_CE signal. Such a result would be intriguing – it might mean that some other parameter (or combination) is more important, or that our definition of channels needs revision.

These falsification tests elevate the pipeline from a pure inference tool to a scientific probe of theory. They ensure that success and failure modes are clearly defined:

- **Success** means the models can explain the data without internal contradictions (epistemic under control) and the ML identifies the expected key physics (e.g., α_CE) as important – thereby lending credence to our current theoretical picture.
- **Failure** in either test is actually a valuable result. It would prompt a re-examination of model assumptions. For example, a failure of Test 1 (epistemic dominance) might push the community to reconcile differences between COMPAS and POSYDON – perhaps updating one or both, or incorporating missing physics until the models agree better. A failure of Test 2 (CE ineffectiveness) might prompt exploring alternative formation scenarios or parameters.

It's worth noting how novel this falsification framework is in the context of ML-for-astrophysics. Many machine learning models will just give you a fit and maybe some uncertainties, but they won't tell you "I should be thrown out if X happens." Here we've built that in by design. It reflects a philosophy akin to hypothesis testing in science. By establishing these criteria upfront, we avoid the trap of over-interpreting results. Instead, the pipeline will explicitly indicate when its own assumptions break down.

---

## Results and Outputs

Assuming the pipeline runs and the model is not falsified outright, we end up with a wealth of results that can be analyzed and exported. These include:

### Parameter Posteriors

The main output is the posterior distribution for each hyperparameter in θ given the real GW data. For example, we might report a posterior on the common-envelope efficiency α_CE peaking at a certain value with a credible interval, or a posterior on the average black hole natal kick velocity. If multiple codes were used, this posterior inherently accounts for their differences (unless Test 1 failed, in which case we'd be cautious). These results can be compared to previous constraints from the literature.

### Formation Channel Probabilities

We will output the inferred branching fractions of each formation channel in the population. For example, the result could be something like: Isolated binaries (classic + CE) contribute 60% (with 90% CI of 40–80%), dynamical channels 30% (10–50%), CHE 10% (0–20%). We will also have per-event channel assignments probabilistically. These can be tabulated for the catalog: e.g., GW190521 has 80% chance dynamical, 20% isolated in our inference; GW200129 has 70% chance isolated (CE subchannel) etc., based on its properties. Such tables provide rigorous probabilistic classification of individual GW events by formation scenario.

### Uncertainty Breakdown

We will present metrics that show how much uncertainty is left and of what type. For instance, a figure might show two pie charts or histograms – one illustrating the contribution of simulator-related uncertainty vs. statistical uncertainty to the error bars on each parameter. Or a plot of mutual information between code predictions and event parameters for each event. We could also output the mutual information value computed in Test 1 and whether it crosses our threshold.

### Falsification Test Results

Explicit statements about the two tests will be given. For example: "Test 1 (Epistemic Dominance): Only 10% of events showed simulator disagreement exceeding statistical error, thus model ensemble is deemed valid (pass). Test 2 (CE Ineffectiveness): The common-envelope efficiency parameter emerged as a key factor (rank correlation 0.7 with channel-separating latent variables), supporting the hypothesis that α_CE drives the difference between the identified channels (pass)."

### Figures

The pipeline will produce a set of plots for visualization:

- **Mass and Spin Distributions:** Plots comparing the observed distribution of m1, m2, and χ_eff (with uncertainties) to the posterior predictive distributions from our model. This can reveal how well the model reproduces features like the lower mass gap, the power-law slope of high masses, or the distribution of effective spins.

- **Redshift Distribution:** A plot of the rate of detections as a function of redshift (or distance), comparing model vs. observations. This checks consistency with the assumed star formation history and cosmology influence.

- **Corner Plot for Hyperparameters:** A multi-dimensional posterior plot (corner plot) showing the joint and marginal distributions of key hyperparameters (like α_CE, natal kick velocity σ_kick, mass transfer efficiency, etc., plus perhaps the channel fractions). This illustrates parameter constraints and correlations.

- **Attention/Importance Visualization:** If feasible, we will visualize the cross-modal attention weights or some measure of feature importance. For instance, a heatmap that for each event (rows) and each hyperparameter (columns) shows the attention score or gradient-based importance. This can highlight which events are strongly informing which parameters.

### Tables for Publication

We will produce tables summarizing numeric results in a form convenient for papers or further analysis:

- A table of inferred model parameters with their median and 90% credible intervals
- A table of channel branching fractions, listing each channel and the inferred percentage of mergers from that channel (with uncertainties)
- A table of per-event probabilities for each channel
- A falsification summary table, perhaps binary pass/fail for each test and any quantitative metrics that led to that conclusion

All these outputs are designed to be reproducible and transparent. The pipeline logs the configurations and will provide the data needed for others to inspect or build upon the results.

---

## Conclusion

This end-to-end pipeline represents a comprehensive integration of astrophysical modeling and machine learning to tackle one of the frontiers of gravitational-wave astrophysics: figuring out how and where binary black holes form. It is built on realistic foundations – using well-established simulation codes (e.g. COMPAS, COSMIC, POSYDON) and proven inference methods (neural density estimation via normalizing flows, which have already been applied to GW data). At the same time, it introduces novel elements that push the state of the art:

- A multi-simulator approach that explicitly captures model uncertainties as part of the inference, rather than treating the simulation as gospel.
- A domain adaptation layer to ensure a fair comparison between simulated and real data, addressing the simulation-reality gap that is often ignored.
- An interpretable neural architecture (with event-level and population-level components and cross-attention) that doesn't just output "black box" results but allows us to probe which physics each event is informing.
- Built-in falsification tests that set clear criteria for success or failure of the astrophysical models against the data, embracing the scientific method fully.

By preserving the provenance of every piece of information and by designing the pipeline to be modular and extensible, we also ensure that this work can be reproduced and expanded by others – for example, incorporating new simulation codes as they become available, or adapting to neutron star merger populations, etc. In a rapidly evolving field, such flexibility is key.

In summary, if the pipeline succeeds, we gain robust answers to the formation-channel question with quantified confidence. If it fails, we gain clear guidance on which assumptions to rethink. Either outcome moves us forward in understanding the violent cosmic laboratories that create gravitational-wave sources. The combination of realism (faithfulness to known physics and data characteristics) and novelty (new techniques to handle uncertainties and complexity) makes this pipeline a cutting-edge tool in the era of gravitational-wave astronomy and a model for how to integrate machine learning with domain physics in a principled way.
