"""
Physics-Informed Deep Learning Architecture

This module implements a sophisticated neural network architecture that:
1. Encodes outputs from multiple population synthesis codes
2. Uses cross-modal attention to identify common physics
3. Performs domain adaptation from simulation to observation
4. Quantifies both epistemic (model) and aleatoric (noise) uncertainties
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np
from typing import Dict, List, Tuple, Optional


class PhysicsInformedEncoder(nn.Module):
    """
    Encode population synthesis outputs into latent physics space
    
    Each ensemble member (COMPAS, COSMIC, POSYDON) gets its own encoder to
    capture code-specific features before fusion.
    
    The encoder outputs mean and log_variance for variational inference,
    enabling uncertainty quantification at the encoding stage.
    """
    
    def __init__(self, input_dim: int = 128, latent_dim: int = 64):
        """
        Args:
            input_dim: Dimension of input features from population synthesis
            latent_dim: Dimension of latent physics representation
        """
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, latent_dim * 2)  # Mean and log_var for VAE
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters
        
        Args:
            x: Input features [batch, input_dim]
            
        Returns:
            mean: Latent mean [batch, latent_dim]
            log_var: Latent log variance [batch, latent_dim]
        """
        h = self.encoder(x)
        mean, log_var = torch.chunk(h, 2, dim=-1)
        return mean, log_var


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention to isolate α_CE as primary driver
    
    This layer identifies which physics parameters drive formation channel
    differences across population synthesis codes. Used to test falsification
    criterion (2): whether cross-modal attention can isolate α_CE.
    
    The attention mechanism allows the model to weight different population
    synthesis outputs based on their relevance to the observed GW data.
    """
    
    def __init__(self, d_model: int = 64, n_heads: int = 8):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
        """
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            d_model, 
            n_heads, 
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model)
        )
        
        # Learnable parameter importance weights for physics parameters
        # This is key for identifying α_CE as the primary driver
        self.param_importance = nn.Parameter(torch.zeros(10))
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        return_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply cross-modal attention
        
        Args:
            query: Query tensor (typically GW observations) [batch, seq_q, d_model]
            key: Key tensor (population synthesis features) [batch, seq_k, d_model]
            value: Value tensor (population synthesis features) [batch, seq_v, d_model]
            return_weights: If True, return attention weights
            
        Returns:
            output: Attended features [batch, seq_q, d_model]
            attn_weights: Optional attention weights [batch, n_heads, seq_q, seq_k]
        """
        # Multi-head attention with residual connection
        attn_out, attn_weights = self.attention(query, key, value)
        x = self.norm1(query + attn_out)
        
        # Feed-forward network with residual connection
        x = self.norm2(x + self.ffn(x))
        
        if return_weights:
            return x, attn_weights
        return x, None


class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient Reversal Layer for Domain Adaptation
    
    This layer passes input forward unchanged, but reverses gradients during
    backpropagation. Used in domain adversarial training to learn features
    that are invariant to the domain (simulation vs. observation).
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float):
        """Store alpha and pass input through"""
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Reverse gradient and scale by alpha"""
        return -ctx.alpha * grad_output, None


class DomainAdaptationLayer(nn.Module):
    """
    Domain adaptation from simulation to observed GW data
    
    Handles distribution shift between synthetic population synthesis outputs
    and real gravitational wave observations using adversarial training.
    
    The goal is to learn features that are predictive of formation channels
    but invariant to whether the data comes from simulation or observation.
    """
    
    def __init__(self, feature_dim: int = 64):
        """
        Args:
            feature_dim: Dimension of input features
        """
        super().__init__()
        
        # Domain classifier distinguishes simulation vs. observation
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)  # Binary: Source (sim) vs Target (obs)
        )
        
        # Feature transformation for domain-invariant representation
        self.feature_transform = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU()
        )
    
    def forward(
        self, 
        features: torch.Tensor, 
        alpha: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply domain adaptation
        
        Args:
            features: Input features [batch, feature_dim]
            alpha: Gradient reversal layer strength (increases during training)
            
        Returns:
            transformed: Domain-invariant features [batch, feature_dim]
            domain_pred: Domain predictions [batch, 2]
        """
        # Transform features to domain-invariant representation
        transformed = self.feature_transform(features)
        
        # Domain classification with gradient reversal
        # This encourages features that fool the domain classifier
        reversed_features = GradientReversalLayer.apply(transformed, alpha)
        domain_pred = self.domain_classifier(reversed_features)
        
        return transformed, domain_pred


class FormationChannelClassifier(nn.Module):
    """
    Final classifier with uncertainty quantification
    
    Classifies gravitational wave events into formation channels while
    quantifying two types of uncertainty:
    1. Aleatoric (data noise): Irreducible uncertainty from measurement noise
    2. Epistemic (model): Reducible uncertainty from model limitations
    
    Formation Channels:
        Channel I: Isolated binary evolution with stable mass transfer
        Channel II: Dynamical formation in dense stellar environments
        Channel III: Hierarchical triple systems
        Channel IV: Common envelope evolution + stable mass transfer
    """
    
    def __init__(self, input_dim: int = 64, n_channels: int = 4):
        """
        Args:
            input_dim: Dimension of input features
            n_channels: Number of formation channels to classify
        """
        super().__init__()
        self.n_channels = n_channels
        
        # Aleatoric uncertainty network (data-dependent noise)
        self.aleatoric_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_channels)  # Per-class uncertainty
        )
        
        # Main classifier with MC Dropout for epistemic uncertainty
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.5),  # High dropout rate for MC sampling
            
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(64, n_channels)
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        n_samples: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Classify with uncertainty quantification
        
        Uses Monte Carlo Dropout to estimate epistemic uncertainty by
        performing multiple stochastic forward passes.
        
        Args:
            x: Input features [batch, input_dim]
            n_samples: Number of MC samples for epistemic uncertainty
            
        Returns:
            mean_pred: Mean predictions [batch, n_channels]
            aleatoric_std: Aleatoric uncertainty [batch, n_channels]
            epistemic_std: Epistemic uncertainty [batch, n_channels]
        """
        # Aleatoric uncertainty (predicted noise)
        aleatoric_std = F.softplus(self.aleatoric_net(x))
        
        # Epistemic uncertainty via MC Dropout
        self.train()  # Enable dropout even during eval
        predictions = []
        
        for _ in range(n_samples):
            logits = self.classifier(x)
            predictions.append(F.softmax(logits, dim=-1))
        
        predictions = torch.stack(predictions)  # [n_samples, batch, n_channels]
        
        # Compute statistics over MC samples
        mean_pred = predictions.mean(dim=0)
        epistemic_std = predictions.std(dim=0)
        
        return mean_pred, aleatoric_std, epistemic_std


class GWObservationEncoder(nn.Module):
    """
    Encode gravitational wave observations
    
    Processes GW parameters (m1, m2, χ_eff, χ_p, z, etc.) into a latent
    representation compatible with population synthesis encodings.
    """
    
    def __init__(self, obs_dim: int = 10, latent_dim: int = 64):
        """
        Args:
            obs_dim: Dimension of GW observables
            latent_dim: Dimension of latent representation
        """
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU()
        )
    
    def forward(self, gw_obs: torch.Tensor) -> torch.Tensor:
        """
        Encode GW observations
        
        Args:
            gw_obs: GW parameters [batch, obs_dim]
            
        Returns:
            encoded: Latent representation [batch, latent_dim]
        """
        return self.encoder(gw_obs)


class PhysicsInformedEnsembleModel(nn.Module):
    """
    Full Physics-Informed Ensemble Architecture
    
    This is the main model that combines:
    1. Multiple population synthesis code encoders
    2. Cross-modal attention for parameter importance
    3. Domain adaptation for sim→obs transfer
    4. Formation channel classification with uncertainty
    
    The model is designed to test two key falsification criteria:
    1. Whether ensemble epistemic uncertainty exceeds observational uncertainty
    2. Whether cross-modal attention can isolate α_CE as the primary driver
    """
    
    def __init__(
        self,
        n_codes: int = 3,  # COMPAS, COSMIC, POSYDON
        input_dim: int = 128,
        latent_dim: int = 64,
        obs_dim: int = 10,
        n_channels: int = 4
    ):
        """
        Args:
            n_codes: Number of population synthesis codes in ensemble
            input_dim: Dimension of pop synth features
            latent_dim: Dimension of latent space
            obs_dim: Dimension of GW observables
            n_channels: Number of formation channels
        """
        super().__init__()
        
        self.n_codes = n_codes
        self.latent_dim = latent_dim
        
        # Separate encoder for each population synthesis code
        self.encoders = nn.ModuleList([
            PhysicsInformedEncoder(input_dim, latent_dim)
            for _ in range(n_codes)
        ])
        
        # Encode gravitational wave observations
        self.gw_encoder = GWObservationEncoder(obs_dim, latent_dim)
        
        # Cross-modal attention to identify common physics
        self.cross_attention = CrossModalAttention(latent_dim)
        
        # Domain adaptation layer
        self.domain_adapter = DomainAdaptationLayer(latent_dim)
        
        # Final classifier with uncertainty quantification
        self.classifier = FormationChannelClassifier(latent_dim, n_channels)
        
        # Learnable physics parameter embeddings (for α_CE correlation)
        self.physics_params = nn.Parameter(torch.randn(10, latent_dim))
    
    def reparameterize(
        self, 
        mean: torch.Tensor, 
        log_var: torch.Tensor
    ) -> torch.Tensor:
        """
        Reparameterization trick for variational inference
        
        z = μ + σ * ε, where ε ~ N(0, 1)
        
        Args:
            mean: Latent mean [batch, latent_dim]
            log_var: Latent log variance [batch, latent_dim]
            
        Returns:
            z: Sampled latent [batch, latent_dim]
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(
        self,
        code_inputs: List[torch.Tensor],
        gw_observations: torch.Tensor,
        training: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through entire architecture
        
        Args:
            code_inputs: List of [batch, input_dim] tensors from each code
            gw_observations: [batch, obs_dim] GW parameters
            training: Whether in training mode (affects domain adaptation)
            
        Returns:
            Dictionary containing:
                - channel_probs: Formation channel probabilities
                - aleatoric_uncertainty: Data noise uncertainty
                - epistemic_uncertainty: Model uncertainty
                - mutual_information: Ensemble disagreement metric
                - attention_weights: For α_CE correlation analysis
                - domain_pred: Domain classification logits
                - kl_divergence: KL divergence for VAE loss
        """
        batch_size = code_inputs[0].shape[0]
        
        # Encode each population synthesis code
        latent_representations = []
        kl_divergences = []
        
        for i, encoder in enumerate(self.encoders):
            mean, log_var = encoder(code_inputs[i])
            
            # Reparameterization trick
            z = self.reparameterize(mean, log_var)
            latent_representations.append(z)
            
            # KL divergence for VAE loss: KL(q(z|x) || p(z))
            # where p(z) = N(0, 1)
            kl_div = -0.5 * torch.sum(
                1 + log_var - mean.pow(2) - log_var.exp(),
                dim=-1
            )
            kl_divergences.append(kl_div)
        
        # Stack latent representations for cross-attention
        latent_stack = torch.stack(
            latent_representations, 
            dim=1
        )  # [batch, n_codes, latent_dim]
        
        # Encode gravitational wave observations
        gw_embedding = self.gw_encoder(gw_observations)  # [batch, latent_dim]
        
        # Cross-modal attention: GW obs attend to pop synth features
        attended_features, attn_weights = self.cross_attention(
            query=gw_embedding.unsqueeze(1),  # [batch, 1, latent_dim]
            key=latent_stack,
            value=latent_stack,
            return_weights=True
        )
        
        attended_features = attended_features.squeeze(1)  # [batch, latent_dim]
        
        # Domain adaptation (if training)
        alpha = 1.0 if training else 0.0
        adapted_features, domain_pred = self.domain_adapter(
            attended_features,
            alpha=alpha
        )
        
        # Classification with uncertainty quantification
        channel_probs, aleatoric_unc, epistemic_unc = self.classifier(
            adapted_features,
            n_samples=20 if not training else 10
        )
        
        # Calculate mutual information for epistemic uncertainty test
        # This measures disagreement between population synthesis codes
        mutual_info = self.calculate_mutual_information(latent_representations)
        
        return {
            'channel_probs': channel_probs,
            'aleatoric_uncertainty': aleatoric_unc,
            'epistemic_uncertainty': epistemic_unc,
            'mutual_information': mutual_info,
            'attention_weights': attn_weights,
            'domain_pred': domain_pred,
            'kl_divergence': torch.stack(kl_divergences).mean(dim=0)
        }
    
    def calculate_mutual_information(
        self, 
        latent_reps: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Calculate mutual information I(Y; C | X)
        
        Measures information shared between ensemble predictions and formation
        channels given GW data. High MI indicates ensemble disagreement.
        
        For falsification criterion (1):
        If MI > observational uncertainty for >50% of events → falsified
        
        Simplified estimation via ensemble variance
        
        Args:
            latent_reps: List of latent representations from each code
            
        Returns:
            mi: Mutual information estimate [batch]
        """
        latent_stack = torch.stack(latent_reps, dim=1)  # [batch, n_codes, latent_dim]
        
        # Ensemble variance as proxy for mutual information
        # Higher variance = more disagreement = higher epistemic uncertainty
        ensemble_variance = latent_stack.var(dim=1).mean(dim=-1)  # [batch]
        
        return ensemble_variance


def build_model(config: Dict) -> PhysicsInformedEnsembleModel:
    """
    Factory function to build model from configuration
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Initialized model
    """
    model = PhysicsInformedEnsembleModel(
        n_codes=config.get('n_codes', 3),
        input_dim=config.get('input_dim', 128),
        latent_dim=config.get('latent_dim', 64),
        obs_dim=config.get('obs_dim', 10),
        n_channels=config.get('n_channels', 4)
    )
    
    return model

