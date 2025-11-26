#!/usr/bin/env python3
"""
Training Script for Physics-Informed Formation Channel Inference

This script trains the complete physics-informed ensemble model for
gravitational wave formation channel inference with uncertainty quantification.
"""

import argparse
import yaml
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Dict, List
from datetime import datetime

# Import project modules
from models.physics_informed_nn import PhysicsInformedEnsembleModel, build_model
from data.gwtc4_loader import GWTC4Loader, create_synthetic_gwtc4_for_testing
from inference.sbi_framework import PopulationSynthesisPrior, SBIFramework
from falsification.test_criteria import FalsificationTester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FormationChannelTrainer:
    """
    Main trainer class for physics-informed formation channel inference
    
    Handles:
        - Model initialization and training
        - Loss computation with multiple objectives
        - Validation and evaluation
        - Checkpointing and logging
    """
    
    def __init__(self, config: Dict):
        """
        Initialize trainer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Set device
        self.device = self._setup_device()
        logger.info(f"Using device: {self.device}")
        
        # Set random seed for reproducibility
        self._set_seed(config['computation']['seed'])
        
        # Initialize model
        self.model = self._build_model()
        
        # Initialize optimizer and scheduler
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        
        # Setup checkpointing
        self.checkpoint_dir = Path(config['training']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup tensorboard
        tensorboard_dir = Path(config['logging']['tensorboard_dir'])
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(tensorboard_dir)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        logger.info("Trainer initialized successfully")
    
    def _setup_device(self) -> torch.device:
        """Setup computation device"""
        device_name = self.config['computation']['device']
        
        if device_name == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        elif device_name == 'mps' and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _build_model(self) -> nn.Module:
        """Build physics-informed ensemble model"""
        model = build_model(self.config['model'])
        model = model.to(self.device)
        
        # Count parameters
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model has {n_params:,} trainable parameters")
        
        return model
    
    def _build_optimizer(self) -> optim.Optimizer:
        """Build optimizer"""
        optimizer_name = self.config['training']['optimizer']
        lr = self.config['training']['learning_rate']
        weight_decay = self.config['training']['weight_decay']
        
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'AdamW':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        return optimizer
    
    def _build_scheduler(self):
        """Build learning rate scheduler"""
        scheduler_name = self.config['training']['scheduler']
        
        if scheduler_name == 'CosineAnnealingLR':
            params = self.config['training']['scheduler_params']
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=params['T_max'],
                eta_min=params['eta_min']
            )
        elif scheduler_name == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10
            )
        else:
            scheduler = None
        
        return scheduler
    
    def compute_loss(
        self,
        output: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        domain_labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-objective loss
        
        Loss components:
            1. Classification loss (cross-entropy)
            2. KL divergence (VAE regularization)
            3. Domain adaptation loss
            4. Aleatoric uncertainty regularization
        
        Args:
            output: Model output dictionary
            targets: True formation channel labels [batch]
            domain_labels: Domain labels (0=sim, 1=obs) [batch]
            
        Returns:
            Dictionary with loss components and total loss
        """
        weights = self.config['training']['loss_weights']
        
        # 1. Classification loss
        channel_probs = output['channel_probs']
        classification_loss = nn.CrossEntropyLoss()(
            channel_probs.log(),
            targets
        )
        
        # 2. KL divergence (VAE regularization)
        kl_loss = output['kl_divergence'].mean()
        
        # 3. Domain adaptation loss
        domain_pred = output['domain_pred']
        domain_loss = nn.CrossEntropyLoss()(domain_pred, domain_labels)
        
        # 4. Aleatoric uncertainty regularization
        # Encourage model to predict uncertainty when needed
        aleatoric_unc = output['aleatoric_uncertainty']
        aleatoric_loss = -aleatoric_unc.mean()  # Negative to encourage exploration
        
        # Total loss
        total_loss = (
            weights['classification'] * classification_loss +
            weights['kl_divergence'] * kl_loss +
            weights['domain_adaptation'] * domain_loss +
            weights['aleatoric'] * aleatoric_loss
        )
        
        return {
            'total': total_loss,
            'classification': classification_loss,
            'kl_divergence': kl_loss,
            'domain_adaptation': domain_loss,
            'aleatoric': aleatoric_loss
        }
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary with average losses
        """
        self.model.train()
        
        total_losses = {
            'total': 0.0,
            'classification': 0.0,
            'kl_divergence': 0.0,
            'domain_adaptation': 0.0,
            'aleatoric': 0.0
        }
        n_batches = 0
        
        for batch_idx, batch_data in enumerate(train_loader):
            # Unpack batch
            # For simplicity, assume batch_data contains:
            # (code_inputs_list, gw_obs, targets, domain_labels)
            code_inputs = [x.to(self.device) for x in batch_data[0]]
            gw_obs = batch_data[1].to(self.device)
            targets = batch_data[2].to(self.device)
            domain_labels = batch_data[3].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(code_inputs, gw_obs, training=True)
            
            # Compute loss
            losses = self.compute_loss(output, targets, domain_labels)
            
            # Backward pass
            losses['total'].backward()
            
            # Gradient clipping
            if self.config['training']['clip_grad_norm'] > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['clip_grad_norm']
                )
            
            self.optimizer.step()
            
            # Accumulate losses
            for key in total_losses:
                total_losses[key] += losses[key].item()
            n_batches += 1
            
            # Log batch progress
            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {self.current_epoch} "
                    f"[{batch_idx}/{len(train_loader)}] "
                    f"Loss: {losses['total'].item():.4f}"
                )
        
        # Average losses
        avg_losses = {k: v / n_batches for k, v in total_losses.items()}
        
        return avg_losses
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate model
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary with average validation losses and metrics
        """
        self.model.eval()
        
        total_losses = {
            'total': 0.0,
            'classification': 0.0,
            'kl_divergence': 0.0,
            'domain_adaptation': 0.0,
            'aleatoric': 0.0
        }
        n_batches = 0
        
        # Accuracy tracking
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                code_inputs = [x.to(self.device) for x in batch_data[0]]
                gw_obs = batch_data[1].to(self.device)
                targets = batch_data[2].to(self.device)
                domain_labels = batch_data[3].to(self.device)
                
                # Forward pass
                output = self.model(code_inputs, gw_obs, training=False)
                
                # Compute loss
                losses = self.compute_loss(output, targets, domain_labels)
                
                # Accumulate losses
                for key in total_losses:
                    total_losses[key] += losses[key].item()
                n_batches += 1
                
                # Calculate accuracy
                predictions = output['channel_probs'].argmax(dim=1)
                correct += (predictions == targets).sum().item()
                total += targets.size(0)
        
        # Average losses and accuracy
        avg_losses = {k: v / n_batches for k, v in total_losses.items()}
        avg_losses['accuracy'] = correct / total if total > 0 else 0.0
        
        return avg_losses
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ):
        """
        Main training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        n_epochs = self.config['training']['n_epochs']
        save_every = self.config['training']['save_every']
        patience = self.config['training']['early_stopping_patience']
        
        logger.info(f"Starting training for {n_epochs} epochs")
        
        for epoch in range(n_epochs):
            self.current_epoch = epoch
            
            logger.info(f"\nEpoch {epoch + 1}/{n_epochs}")
            logger.info("=" * 60)
            
            # Train
            train_losses = self.train_epoch(train_loader)
            
            # Validate
            val_losses = self.validate(val_loader)
            
            # Log to tensorboard
            for key, value in train_losses.items():
                self.writer.add_scalar(f'Train/{key}', value, epoch)
            for key, value in val_losses.items():
                self.writer.add_scalar(f'Val/{key}', value, epoch)
            
            # Log to console
            logger.info(f"Train Loss: {train_losses['total']:.4f}")
            logger.info(f"Val Loss: {val_losses['total']:.4f}")
            logger.info(f"Val Accuracy: {val_losses['accuracy']:.2%}")
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_losses['total'])
                else:
                    self.scheduler.step()
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')
            
            # Early stopping
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.patience_counter = 0
                self.save_checkpoint('best_model.pth')
                logger.info("Saved best model")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    logger.info(f"Early stopping after {epoch + 1} epochs")
                    break
        
        logger.info("\nTraining completed!")
        self.writer.close()
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint_path = self.checkpoint_dir / filename
        
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }, checkpoint_path)
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dummy_dataloaders(config: Dict, device: str):
    """
    Create dummy dataloaders for testing
    
    In production, this would load actual COMPAS ensemble and GWTC-4 data
    """
    batch_size = config['training']['batch_size']
    n_codes = config['model']['n_codes']
    input_dim = config['model']['input_dim']
    obs_dim = config['model']['obs_dim']
    
    # Generate dummy data
    n_train = 1000
    n_val = 200
    
    # Training data
    train_code_inputs = [torch.randn(n_train, input_dim) for _ in range(n_codes)]
    train_gw_obs = torch.randn(n_train, obs_dim)
    train_targets = torch.randint(0, 4, (n_train,))
    train_domain = torch.zeros(n_train, dtype=torch.long)  # All simulation
    
    train_dataset = TensorDataset(
        *train_code_inputs,
        train_gw_obs,
        train_targets,
        train_domain
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Validation data
    val_code_inputs = [torch.randn(n_val, input_dim) for _ in range(n_codes)]
    val_gw_obs = torch.randn(n_val, obs_dim)
    val_targets = torch.randint(0, 4, (n_val,))
    val_domain = torch.zeros(n_val, dtype=torch.long)
    
    val_dataset = TensorDataset(
        *val_code_inputs,
        val_gw_obs,
        val_targets,
        val_domain
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Wrap loaders to return expected format
    def wrap_loader(loader, n_codes):
        for batch in loader:
            code_inputs = [batch[i] for i in range(n_codes)]
            gw_obs = batch[n_codes]
            targets = batch[n_codes + 1]
            domain = batch[n_codes + 2]
            yield (code_inputs, gw_obs, targets, domain)
    
    train_loader_wrapped = list(wrap_loader(train_loader, n_codes))
    val_loader_wrapped = list(wrap_loader(val_loader, n_codes))
    
    return train_loader_wrapped, val_loader_wrapped


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(
        description='Train physics-informed formation channel inference model'
    )
    parser.add_argument(
        '--config',
        default='configs/default_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--checkpoint',
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Run in test mode with dummy data'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Initialize trainer
    trainer = FormationChannelTrainer(config)
    
    # Load checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # Create dataloaders
    if args.test_mode:
        logger.info("Running in TEST MODE with dummy data")
        train_loader, val_loader = create_dummy_dataloaders(
            config,
            trainer.device
        )
    else:
        # Load real data (to be implemented)
        logger.warning("Real data loading not yet implemented, using dummy data")
        train_loader, val_loader = create_dummy_dataloaders(
            config,
            trainer.device
        )
    
    # Train model
    trainer.train(train_loader, val_loader)
    
    logger.info("Training script completed successfully!")


if __name__ == "__main__":
    main()

