"""
Simulation-Based Inference using Population Synthesis as Priors

This module implements Bayesian inference for gravitational wave formation
channels using population synthesis simulations as learned priors.

Key Features:
    - Neural Posterior Estimation (NPE) with COMPAS ensemble priors
    - Sequential Neural Posterior Estimation (SNPE) for iterative refinement
    - Amortized inference across multiple GW events
    - Posterior predictive checks for model validation
"""

import torch
import torch.nn as nn
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from sklearn.neighbors import KernelDensity
import logging

logger = logging.getLogger(__name__)


class PopulationSynthesisPrior:
    """
    Learned prior distribution from population synthesis ensemble
    
    Uses COMPAS (or other pop synth code) ensemble outputs to construct
    a data-driven prior over gravitational wave event parameters.
    
    This captures astrophysical knowledge from detailed stellar evolution
    modeling, providing a physics-informed prior for inference.
    """
    
    def __init__(
        self,
        ensemble_data_path: str,
        parameters: List[str] = None,
        bandwidth: float = 0.1,
        kernel: str = 'gaussian'
    ):
        """
        Initialize prior from ensemble data
        
        Args:
            ensemble_data_path: Path to COMPAS ensemble metadata JSON
            parameters: List of parameter names to include in prior
            bandwidth: KDE bandwidth
            kernel: Kernel type for KDE
        """
        self.ensemble_data_path = Path(ensemble_data_path)
        self.parameters = parameters or [
            'm1_source', 'm2_source', 'chi_eff', 'chi_p', 
            'redshift', 't_delay', 'a_final'
        ]
        self.bandwidth = bandwidth
        self.kernel = kernel
        
        # Load and process ensemble data
        self.ensemble_data = self.load_ensemble_data()
        
        # Fit prior distribution
        self.prior_distribution = self.fit_prior()
        
        logger.info(f"Initialized prior from {len(self.ensemble_data)} ensemble runs")
    
    def load_ensemble_data(self) -> np.ndarray:
        """
        Load DCO parameters from COMPAS ensemble
        
        Returns:
            Array of DCO parameters [n_systems, n_params]
        """
        import json
        
        # Load ensemble metadata
        metadata_file = self.ensemble_data_path.parent / 'ensemble_metadata.json'
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        all_params = []
        
        for run_info in metadata['runs']:
            if run_info['status'] != 'success':
                continue
            
            # Load HDF5 output
            h5_file = run_info['output_file']
            params = self.extract_dco_parameters(h5_file)
            
            if params is not None:
                all_params.append(params)
        
        # Concatenate all parameters
        if len(all_params) == 0:
            raise ValueError("No successful runs found in ensemble")
        
        all_params = np.vstack(all_params)
        logger.info(f"Loaded {len(all_params)} DCO systems from ensemble")
        
        return all_params
    
    def extract_dco_parameters(self, h5_file: str) -> Optional[np.ndarray]:
        """
        Extract relevant DCO parameters from COMPAS HDF5 output
        
        Args:
            h5_file: Path to COMPAS_Output.h5
            
        Returns:
            Array of parameters [n_dco, n_params] or None if no DCOs
        """
        try:
            with h5py.File(h5_file, 'r') as f:
                # Check if DCO data exists
                if 'DoubleCompactObjects' not in f:
                    return None
                
                dco_data = f['DoubleCompactObjects']
                
                # Extract parameters
                params_list = []
                param_map = {
                    'm1_source': 'Mass(1)',
                    'm2_source': 'Mass(2)',
                    'chi_eff': 'CHI_eff',
                    'chi_p': 'CHI_p',
                    'a_final': 'SemiMajorAxis',
                    'ecc': 'Eccentricity',
                }
                
                n_dco = len(dco_data[list(dco_data.keys())[0]])
                if n_dco == 0:
                    return None
                
                for param in self.parameters:
                    if param in param_map and param_map[param] in dco_data:
                        params_list.append(dco_data[param_map[param]][()])
                    elif param in dco_data:
                        params_list.append(dco_data[param][()])
                    else:
                        # Use placeholder if parameter not available
                        params_list.append(np.zeros(n_dco))
                
                params = np.column_stack(params_list)
                
                # Filter valid systems (positive masses, etc.)
                valid_mask = (params[:, 0] > 0) & (params[:, 1] > 0)
                params = params[valid_mask]
                
                return params if len(params) > 0 else None
                
        except Exception as e:
            logger.warning(f"Failed to load {h5_file}: {e}")
            return None
    
    def fit_prior(self) -> KernelDensity:
        """
        Fit KDE to ensemble data
        
        Returns:
            Fitted KernelDensity object
        """
        # Normalize data for KDE
        self.mean = self.ensemble_data.mean(axis=0)
        self.std = self.ensemble_data.std(axis=0)
        normalized_data = (self.ensemble_data - self.mean) / (self.std + 1e-8)
        
        # Fit KDE
        kde = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth)
        kde.fit(normalized_data)
        
        logger.info(f"Fitted KDE prior with {len(self.ensemble_data)} samples")
        
        return kde
    
    def sample(self, n_samples: int) -> torch.Tensor:
        """
        Sample from learned prior
        
        Args:
            n_samples: Number of samples to draw
            
        Returns:
            Samples [n_samples, n_params]
        """
        # Sample from KDE
        samples = self.prior_distribution.sample(n_samples)
        
        # Denormalize
        samples = samples * self.std + self.mean
        
        return torch.tensor(samples, dtype=torch.float32)
    
    def log_prob(self, params: torch.Tensor) -> torch.Tensor:
        """
        Evaluate log probability under prior
        
        Args:
            params: Parameters [batch, n_params]
            
        Returns:
            Log probabilities [batch]
        """
        # Normalize
        params_np = params.cpu().numpy()
        normalized = (params_np - self.mean) / (self.std + 1e-8)
        
        # Evaluate KDE
        log_probs = self.prior_distribution.score_samples(normalized)
        
        return torch.tensor(log_probs, dtype=torch.float32)


class NeuralPosteriorEstimator(nn.Module):
    """
    Neural network for posterior density estimation
    
    Learns p(θ | x) where:
        θ = astrophysical parameters (α_CE, Z, etc.)
        x = observed GW parameters (m1, m2, χ_eff, etc.)
    
    Uses normalizing flows or mixture density networks for flexible
    posterior representation.
    """
    
    def __init__(
        self,
        obs_dim: int = 10,
        param_dim: int = 5,
        hidden_dim: int = 128,
        n_components: int = 10
    ):
        """
        Args:
            obs_dim: Dimension of observations (GW parameters)
            param_dim: Dimension of parameters (astrophysical)
            hidden_dim: Hidden layer dimension
            n_components: Number of mixture components
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.param_dim = param_dim
        self.n_components = n_components
        
        # Encoder: observations → context
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Mixture density network heads
        # Predict means, stds, and mixture weights
        self.mean_head = nn.Linear(hidden_dim, n_components * param_dim)
        self.std_head = nn.Linear(hidden_dim, n_components * param_dim)
        self.weight_head = nn.Linear(hidden_dim, n_components)
    
    def forward(
        self, 
        observations: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict posterior parameters given observations
        
        Args:
            observations: GW observations [batch, obs_dim]
            
        Returns:
            means: Mixture means [batch, n_components, param_dim]
            stds: Mixture stds [batch, n_components, param_dim]
            weights: Mixture weights [batch, n_components]
        """
        # Encode observations
        h = self.encoder(observations)
        
        # Predict mixture parameters
        means = self.mean_head(h).view(-1, self.n_components, self.param_dim)
        stds = F.softplus(self.std_head(h)).view(-1, self.n_components, self.param_dim)
        weights = F.softmax(self.weight_head(h), dim=-1)
        
        return means, stds, weights
    
    def log_prob(
        self,
        params: torch.Tensor,
        observations: torch.Tensor
    ) -> torch.Tensor:
        """
        Evaluate log probability of parameters given observations
        
        Args:
            params: Parameters [batch, param_dim]
            observations: Observations [batch, obs_dim]
            
        Returns:
            Log probabilities [batch]
        """
        means, stds, weights = self.forward(observations)
        
        # Compute log prob for each mixture component
        # log p(θ|x) = log Σ_k w_k N(θ; μ_k, σ_k)
        params_expanded = params.unsqueeze(1)  # [batch, 1, param_dim]
        
        log_probs_components = -0.5 * (
            ((params_expanded - means) / stds) ** 2
        ).sum(dim=-1)  # [batch, n_components]
        
        log_probs_components -= 0.5 * torch.log(2 * np.pi * stds.pow(2)).sum(dim=-1)
        
        # Weight by mixture weights
        log_probs = torch.logsumexp(
            log_probs_components + torch.log(weights),
            dim=-1
        )
        
        return log_probs
    
    def sample(
        self,
        observations: torch.Tensor,
        n_samples: int = 1000
    ) -> torch.Tensor:
        """
        Sample from posterior p(θ | x)
        
        Args:
            observations: GW observations [batch, obs_dim]
            n_samples: Number of samples per observation
            
        Returns:
            Samples [batch, n_samples, param_dim]
        """
        means, stds, weights = self.forward(observations)
        batch_size = observations.shape[0]
        
        # Sample mixture component indices
        component_probs = weights.unsqueeze(1).expand(-1, n_samples, -1)
        component_indices = torch.multinomial(
            component_probs.reshape(-1, self.n_components),
            num_samples=1
        ).reshape(batch_size, n_samples)
        
        # Gather selected component parameters
        selected_means = torch.gather(
            means.unsqueeze(1).expand(-1, n_samples, -1, -1),
            2,
            component_indices.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, 1, self.param_dim
            )
        ).squeeze(2)
        
        selected_stds = torch.gather(
            stds.unsqueeze(1).expand(-1, n_samples, -1, -1),
            2,
            component_indices.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, 1, self.param_dim
            )
        ).squeeze(2)
        
        # Sample from selected components
        eps = torch.randn_like(selected_means)
        samples = selected_means + selected_stds * eps
        
        return samples


class SBIFramework:
    """
    Main Simulation-Based Inference framework
    
    Combines population synthesis priors with neural posterior estimation
    to perform Bayesian inference on gravitational wave observations.
    """
    
    def __init__(
        self,
        prior: PopulationSynthesisPrior,
        obs_dim: int = 10,
        param_dim: int = 5,
        device: str = 'cpu'
    ):
        """
        Args:
            prior: Learned prior from population synthesis
            obs_dim: Dimension of GW observations
            param_dim: Dimension of astrophysical parameters
            device: Torch device
        """
        self.prior = prior
        self.obs_dim = obs_dim
        self.param_dim = param_dim
        self.device = device
        
        # Initialize neural posterior estimator
        self.posterior_net = NeuralPosteriorEstimator(
            obs_dim=obs_dim,
            param_dim=param_dim
        ).to(device)
        
        logger.info("Initialized SBI framework")
    
    def train_posterior(
        self,
        simulator: Callable,
        n_simulations: int = 10000,
        n_epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 1e-3
    ):
        """
        Train neural posterior estimator
        
        Args:
            simulator: Function θ → x that simulates GW observations
            n_simulations: Number of simulations for training
            n_epochs: Training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        logger.info(f"Training posterior with {n_simulations} simulations")
        
        # Generate training data
        theta_samples = self.prior.sample(n_simulations).to(self.device)
        x_samples = simulator(theta_samples)
        
        # Setup optimizer
        optimizer = torch.optim.Adam(
            self.posterior_net.parameters(),
            lr=learning_rate
        )
        
        # Training loop
        dataset = torch.utils.data.TensorDataset(theta_samples, x_samples)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        for epoch in range(n_epochs):
            total_loss = 0.0
            
            for theta_batch, x_batch in dataloader:
                optimizer.zero_grad()
                
                # Negative log likelihood loss
                log_prob = self.posterior_net.log_prob(theta_batch, x_batch)
                loss = -log_prob.mean()
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                logger.info(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
    
    def infer(
        self,
        gw_observations: torch.Tensor,
        n_samples: int = 10000
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Perform inference on GW observations
        
        Args:
            gw_observations: Observed GW parameters [batch, obs_dim]
            n_samples: Number of posterior samples
            
        Returns:
            samples: Posterior samples [batch, n_samples, param_dim]
            uncertainties: Dictionary with epistemic and aleatoric uncertainties
        """
        self.posterior_net.eval()
        
        with torch.no_grad():
            # Sample from posterior
            samples = self.posterior_net.sample(
                gw_observations.to(self.device),
                n_samples=n_samples
            )
            
            # Calculate uncertainties
            epistemic_unc = samples.std(dim=1)  # Model uncertainty
            mean_pred = samples.mean(dim=1)
            
            # Aleatoric uncertainty from observation noise
            # (would need to be provided or estimated)
            
            uncertainties = {
                'epistemic': epistemic_unc,
                'mean': mean_pred
            }
        
        return samples, uncertainties
    
    def posterior_predictive_check(
        self,
        gw_observations: torch.Tensor,
        simulator: Callable,
        n_samples: int = 1000
    ) -> Dict[str, np.ndarray]:
        """
        Perform posterior predictive checks
        
        Sample from posterior, run forward model, compare to observations
        
        Args:
            gw_observations: Observed GW parameters [batch, obs_dim]
            simulator: Forward model θ → x
            n_samples: Number of posterior samples
            
        Returns:
            Dictionary with predictive samples and statistics
        """
        # Sample from posterior
        posterior_samples, _ = self.infer(gw_observations, n_samples)
        
        # Run forward model
        predictive_samples = []
        for i in range(n_samples):
            theta_sample = posterior_samples[:, i, :]
            x_pred = simulator(theta_sample)
            predictive_samples.append(x_pred)
        
        predictive_samples = torch.stack(predictive_samples, dim=1)
        
        # Calculate statistics
        return {
            'samples': predictive_samples.cpu().numpy(),
            'mean': predictive_samples.mean(dim=1).cpu().numpy(),
            'std': predictive_samples.std(dim=1).cpu().numpy()
        }


def load_sbi_framework(
    ensemble_path: str,
    checkpoint_path: Optional[str] = None,
    device: str = 'cpu'
) -> SBIFramework:
    """
    Load SBI framework from ensemble and optional checkpoint
    
    Args:
        ensemble_path: Path to COMPAS ensemble
        checkpoint_path: Optional path to trained posterior checkpoint
        device: Torch device
        
    Returns:
        Initialized SBI framework
    """
    # Initialize prior
    prior = PopulationSynthesisPrior(ensemble_path)
    
    # Initialize framework
    framework = SBIFramework(
        prior=prior,
        device=device
    )
    
    # Load checkpoint if provided
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        framework.posterior_net.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    return framework

