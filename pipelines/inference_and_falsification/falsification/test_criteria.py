"""
Falsification Criteria Testing for Formation Channel Inference

This module implements systematic tests to determine when formation channel
inference claims are falsified by the data.

Two key falsification criteria:

Criterion 1: Ensemble Epistemic Uncertainty > Observational Uncertainty
    If epistemic uncertainty from ensemble disagreement exceeds measurement
    uncertainty for >50% of GWTC-4 events, the inference claim is falsified.
    This indicates the models disagree too much to make reliable inferences.

Criterion 2: Cross-Modal Attention Fails to Isolate α_CE
    If rank correlation between α_CE attention weights and Channel I/IV
    assignment is <0.5, the inference is falsified. This indicates α_CE
    is not the primary driver of the formation channel degeneracy.
"""

import torch
import numpy as np
from scipy.stats import spearmanr, kendalltau
from typing import Dict, List, Tuple
import logging
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)


class FalsificationTester:
    """
    Main class for testing falsification criteria on GWTC-4 catalog
    
    Systematically evaluates when formation channel inference claims
    should be rejected based on uncertainty quantification and parameter
    importance analysis.
    """
    
    def __init__(
        self,
        model,
        gwtc4_catalog: List[Dict],
        output_dir: str = './results/tables/falsification'
    ):
        """
        Initialize falsification tester
        
        Args:
            model: Trained physics-informed ensemble model
            gwtc4_catalog: List of GWTC-4 events with observations
            output_dir: Directory for saving results
        """
        self.model = model
        self.gwtc4_catalog = gwtc4_catalog
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {
            'criterion_1': None,
            'criterion_2': None,
            'per_event_results': []
        }
        
        logger.info(f"Initialized falsification tester with {len(gwtc4_catalog)} events")
    
    def test_criterion_1(
        self,
        threshold: float = 0.5,
        save_plots: bool = True
    ) -> Dict:
        """
        Criterion 1: Ensemble epistemic uncertainty > observational uncertainty
        
        Tests whether model uncertainty from ensemble disagreement exceeds
        the irreducible measurement uncertainty from detector noise.
        
        Falsification Rule:
            If epistemic_unc > obs_unc for more than `threshold` fraction
            of events, the inference approach is FALSIFIED.
        
        Args:
            threshold: Falsification threshold (default 0.5 = 50% of events)
            save_plots: Whether to save diagnostic plots
            
        Returns:
            Dictionary with falsification results and statistics
        """
        logger.info("="*60)
        logger.info("TESTING CRITERION 1: Epistemic vs Observational Uncertainty")
        logger.info("="*60)
        
        n_events = len(self.gwtc4_catalog)
        falsification_count = 0
        
        results_list = []
        epistemic_uncs = []
        obs_uncs = []
        event_names = []
        
        self.model.eval()
        
        for i, event in enumerate(self.gwtc4_catalog):
            event_name = event['name']
            logger.info(f"Processing event {i+1}/{n_events}: {event_name}")
            
            # Prepare model inputs
            code_inputs = event['pop_synth_inputs']  # List of tensors
            gw_observations = event['gw_observations']  # Tensor
            
            # Get model predictions
            with torch.no_grad():
                output = self.model(
                    code_inputs,
                    gw_observations.unsqueeze(0),
                    training=False
                )
            
            # Extract epistemic uncertainty (from mutual information)
            # This measures disagreement between ensemble codes
            epistemic_unc = output['mutual_information'].item()
            
            # Observational uncertainty (from detector noise)
            # This is provided in the event data (e.g., from posterior width)
            obs_unc = event['observational_uncertainty']
            
            # Test falsification criterion
            is_falsified = epistemic_unc > obs_unc
            falsification_count += int(is_falsified)
            
            # Store results
            result = {
                'event_name': event_name,
                'epistemic_unc': epistemic_unc,
                'obs_unc': obs_unc,
                'ratio': epistemic_unc / obs_unc if obs_unc > 0 else np.inf,
                'falsified': is_falsified,
                'channel_probs': output['channel_probs'].cpu().numpy()[0],
                'aleatoric_unc': output['aleatoric_uncertainty'].cpu().numpy()[0],
                'epistemic_unc_full': output['epistemic_uncertainty'].cpu().numpy()[0]
            }
            results_list.append(result)
            
            epistemic_uncs.append(epistemic_unc)
            obs_uncs.append(obs_unc)
            event_names.append(event_name)
        
        # Calculate falsification rate
        falsification_rate = falsification_count / n_events
        is_falsified = falsification_rate > threshold
        
        # Summary statistics
        epistemic_uncs = np.array(epistemic_uncs)
        obs_uncs = np.array(obs_uncs)
        ratios = epistemic_uncs / (obs_uncs + 1e-10)
        
        summary = {
            'falsified': is_falsified,
            'falsification_rate': falsification_rate,
            'threshold': threshold,
            'n_events': n_events,
            'n_falsified': falsification_count,
            'mean_epistemic_unc': float(epistemic_uncs.mean()),
            'median_epistemic_unc': float(np.median(epistemic_uncs)),
            'mean_obs_unc': float(obs_uncs.mean()),
            'median_obs_unc': float(np.median(obs_uncs)),
            'mean_ratio': float(ratios.mean()),
            'median_ratio': float(np.median(ratios)),
            'results': results_list
        }
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("CRITERION 1 RESULTS")
        logger.info("="*60)
        logger.info(f"Falsified: {is_falsified}")
        logger.info(f"Falsification rate: {falsification_rate:.2%}")
        logger.info(f"Events with epistemic > obs: {falsification_count}/{n_events}")
        logger.info(f"Mean epistemic uncertainty: {summary['mean_epistemic_unc']:.4f}")
        logger.info(f"Mean observational uncertainty: {summary['mean_obs_unc']:.4f}")
        logger.info(f"Mean ratio (epistemic/obs): {summary['mean_ratio']:.2f}")
        logger.info("="*60 + "\n")
        
        # Save plots
        if save_plots:
            self.plot_criterion_1_results(
                epistemic_uncs,
                obs_uncs,
                event_names,
                is_falsified
            )
        
        # Save results to CSV
        df = pd.DataFrame(results_list)
        csv_path = self.output_dir / 'criterion_1_results.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved results to {csv_path}")
        
        self.results['criterion_1'] = summary
        return summary
    
    def test_criterion_2(
        self,
        min_correlation: float = 0.5,
        save_plots: bool = True
    ) -> Dict:
        """
        Criterion 2: Cross-modal attention fails to isolate α_CE
        
        Tests whether the model's attention mechanism correctly identifies
        α_CE (common envelope efficiency) as the primary parameter driving
        the degeneracy between Channel I and Channel IV formation pathways.
        
        Falsification Rule:
            If rank correlation between α_CE attention weights and Channel I/IV
            classification is < `min_correlation`, the inference is FALSIFIED.
            
        This indicates the model is not correctly identifying the key physics.
        
        Args:
            min_correlation: Minimum required rank correlation (default 0.5)
            save_plots: Whether to save diagnostic plots
            
        Returns:
            Dictionary with falsification results and correlation statistics
        """
        logger.info("="*60)
        logger.info("TESTING CRITERION 2: α_CE Rank Correlation")
        logger.info("="*60)
        
        n_events = len(self.gwtc4_catalog)
        
        alpha_ce_attentions = []
        channel_assignments = []
        channel_probs_list = []
        event_names = []
        
        self.model.eval()
        
        # Extract α_CE attention weights and channel assignments
        for i, event in enumerate(self.gwtc4_catalog):
            event_name = event['name']
            logger.info(f"Processing event {i+1}/{n_events}: {event_name}")
            
            # Prepare model inputs
            code_inputs = event['pop_synth_inputs']
            gw_observations = event['gw_observations']
            
            # Get model predictions with attention weights
            with torch.no_grad():
                output = self.model(
                    code_inputs,
                    gw_observations.unsqueeze(0),
                    training=False
                )
            
            # Extract attention weights for α_CE parameter
            # attn_weights shape: [batch, n_heads, seq_len_q, seq_len_k]
            attn_weights = output['attention_weights']
            
            # Average over heads and queries
            attn_weights_avg = attn_weights.mean(dim=(0, 1))  # [seq_len_k]
            
            # α_CE is the first parameter (index 0) in our parameter space
            alpha_ce_idx = 0
            alpha_ce_attn = attn_weights_avg[alpha_ce_idx].item()
            
            # Get predicted formation channel
            channel_probs = output['channel_probs'].cpu().numpy()[0]
            channel_pred = channel_probs.argmax()
            
            # Focus on Channel I (0) vs Channel IV (3) degeneracy
            # Map to a score: 0 = Channel I, 1 = Channel IV, 0.5 = ambiguous
            if channel_pred == 0:
                channel_score = 0.0
            elif channel_pred == 3:
                channel_score = 1.0
            else:
                # For other channels, use probability weighted score
                channel_score = channel_probs[3] / (channel_probs[0] + channel_probs[3] + 1e-10)
            
            alpha_ce_attentions.append(alpha_ce_attn)
            channel_assignments.append(channel_score)
            channel_probs_list.append(channel_probs)
            event_names.append(event_name)
        
        # Calculate rank correlations
        alpha_ce_attentions = np.array(alpha_ce_attentions)
        channel_assignments = np.array(channel_assignments)
        
        # Spearman rank correlation
        spearman_corr, spearman_p = spearmanr(
            alpha_ce_attentions,
            channel_assignments
        )
        
        # Kendall tau correlation (alternative measure)
        kendall_corr, kendall_p = kendalltau(
            alpha_ce_attentions,
            channel_assignments
        )
        
        # Test falsification criterion
        is_falsified = abs(spearman_corr) < min_correlation
        
        summary = {
            'falsified': is_falsified,
            'spearman_correlation': float(spearman_corr),
            'spearman_p_value': float(spearman_p),
            'kendall_correlation': float(kendall_corr),
            'kendall_p_value': float(kendall_p),
            'min_correlation_threshold': min_correlation,
            'n_events': n_events,
            'alpha_ce_attentions': alpha_ce_attentions.tolist(),
            'channel_assignments': channel_assignments.tolist(),
            'event_names': event_names
        }
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("CRITERION 2 RESULTS")
        logger.info("="*60)
        logger.info(f"Falsified: {is_falsified}")
        logger.info(f"Spearman correlation: {spearman_corr:.3f} (p={spearman_p:.3e})")
        logger.info(f"Kendall tau: {kendall_corr:.3f} (p={kendall_p:.3e})")
        logger.info(f"Minimum required: {min_correlation:.3f}")
        logger.info(f"Mean α_CE attention: {alpha_ce_attentions.mean():.3f}")
        logger.info(f"Std α_CE attention: {alpha_ce_attentions.std():.3f}")
        logger.info("="*60 + "\n")
        
        # Save plots
        if save_plots:
            self.plot_criterion_2_results(
                alpha_ce_attentions,
                channel_assignments,
                event_names,
                spearman_corr,
                is_falsified
            )
        
        # Save results
        df = pd.DataFrame({
            'event_name': event_names,
            'alpha_ce_attention': alpha_ce_attentions,
            'channel_score': channel_assignments
        })
        csv_path = self.output_dir / 'criterion_2_results.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved results to {csv_path}")
        
        self.results['criterion_2'] = summary
        return summary
    
    def run_all_tests(
        self,
        criterion_1_threshold: float = 0.5,
        criterion_2_min_corr: float = 0.5,
        save_plots: bool = True
    ) -> Dict:
        """
        Run both falsification criteria tests
        
        Args:
            criterion_1_threshold: Threshold for Criterion 1
            criterion_2_min_corr: Minimum correlation for Criterion 2
            save_plots: Whether to save diagnostic plots
            
        Returns:
            Dictionary with combined results
        """
        logger.info("="*60)
        logger.info("RUNNING FULL FALSIFICATION TEST SUITE")
        logger.info("="*60)
        
        # Test Criterion 1
        result_1 = self.test_criterion_1(
            threshold=criterion_1_threshold,
            save_plots=save_plots
        )
        
        # Test Criterion 2
        result_2 = self.test_criterion_2(
            min_correlation=criterion_2_min_corr,
            save_plots=save_plots
        )
        
        # Combined summary
        combined = {
            'criterion_1': result_1,
            'criterion_2': result_2,
            'overall_falsified': result_1['falsified'] or result_2['falsified']
        }
        
        # Print final summary
        logger.info("\n" + "="*60)
        logger.info("FALSIFICATION TEST SUITE SUMMARY")
        logger.info("="*60)
        logger.info(f"Criterion 1 (Epistemic > Obs): {'FALSIFIED' if result_1['falsified'] else 'PASSED'}")
        logger.info(f"Criterion 2 (α_CE Correlation): {'FALSIFIED' if result_2['falsified'] else 'PASSED'}")
        logger.info(f"Overall: {'FALSIFIED' if combined['overall_falsified'] else 'PASSED'}")
        logger.info("="*60 + "\n")
        
        # Save combined results
        import json
        json_path = self.output_dir / 'falsification_summary.json'
        with open(json_path, 'w') as f:
            json.dump(combined, f, indent=2)
        logger.info(f"Saved summary to {json_path}")
        
        return combined
    
    def plot_criterion_1_results(
        self,
        epistemic_uncs: np.ndarray,
        obs_uncs: np.ndarray,
        event_names: List[str],
        is_falsified: bool
    ):
        """Create diagnostic plots for Criterion 1"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Scatter plot
        ax = axes[0, 0]
        ax.scatter(obs_uncs, epistemic_uncs, alpha=0.6, s=50)
        ax.plot([0, max(obs_uncs)], [0, max(obs_uncs)], 'r--', label='1:1 line')
        ax.set_xlabel('Observational Uncertainty', fontsize=12)
        ax.set_ylabel('Epistemic Uncertainty', fontsize=12)
        ax.set_title('Epistemic vs Observational Uncertainty', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Ratio distribution
        ax = axes[0, 1]
        ratios = epistemic_uncs / (obs_uncs + 1e-10)
        ax.hist(ratios, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(1.0, color='r', linestyle='--', linewidth=2, label='Threshold')
        ax.set_xlabel('Epistemic / Observational Ratio', fontsize=12)
        ax.set_ylabel('Number of Events', fontsize=12)
        ax.set_title('Uncertainty Ratio Distribution', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Individual events
        ax = axes[1, 0]
        x = np.arange(len(event_names))
        width = 0.35
        ax.bar(x - width/2, epistemic_uncs, width, label='Epistemic', alpha=0.7)
        ax.bar(x + width/2, obs_uncs, width, label='Observational', alpha=0.7)
        ax.set_xlabel('Event Index', fontsize=12)
        ax.set_ylabel('Uncertainty', fontsize=12)
        ax.set_title('Per-Event Uncertainties', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Cumulative distribution
        ax = axes[1, 1]
        sorted_ratios = np.sort(ratios)
        cumulative = np.arange(1, len(sorted_ratios) + 1) / len(sorted_ratios)
        ax.plot(sorted_ratios, cumulative, linewidth=2)
        ax.axvline(1.0, color='r', linestyle='--', linewidth=2, label='Threshold')
        ax.axhline(0.5, color='g', linestyle='--', linewidth=2, alpha=0.5, label='50%')
        ax.set_xlabel('Epistemic / Observational Ratio', fontsize=12)
        ax.set_ylabel('Cumulative Fraction', fontsize=12)
        ax.set_title('Cumulative Distribution', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Overall title
        status = "FALSIFIED" if is_falsified else "PASSED"
        fig.suptitle(
            f'Criterion 1: Epistemic vs Observational Uncertainty - {status}',
            fontsize=16,
            fontweight='bold'
        )
        
        plt.tight_layout()
        plot_path = self.output_dir / 'criterion_1_plots.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plots to {plot_path}")
        plt.close()
    
    def plot_criterion_2_results(
        self,
        alpha_ce_attentions: np.ndarray,
        channel_assignments: np.ndarray,
        event_names: List[str],
        correlation: float,
        is_falsified: bool
    ):
        """Create diagnostic plots for Criterion 2"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Scatter with correlation
        ax = axes[0]
        ax.scatter(alpha_ce_attentions, channel_assignments, alpha=0.6, s=80)
        
        # Fit line
        z = np.polyfit(alpha_ce_attentions, channel_assignments, 1)
        p = np.poly1d(z)
        x_line = np.linspace(alpha_ce_attentions.min(), alpha_ce_attentions.max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.7)
        
        ax.set_xlabel('α_CE Attention Weight', fontsize=12)
        ax.set_ylabel('Channel Score (0=I, 1=IV)', fontsize=12)
        ax.set_title(f'α_CE Attention vs Channel Assignment\nρ={correlation:.3f}', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Attention distribution
        ax = axes[1]
        ax.hist(alpha_ce_attentions, bins=20, alpha=0.7, edgecolor='black')
        ax.axvline(alpha_ce_attentions.mean(), color='r', linestyle='--', 
                   linewidth=2, label=f'Mean={alpha_ce_attentions.mean():.3f}')
        ax.set_xlabel('α_CE Attention Weight', fontsize=12)
        ax.set_ylabel('Number of Events', fontsize=12)
        ax.set_title('α_CE Attention Distribution', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Overall title
        status = "FALSIFIED" if is_falsified else "PASSED"
        fig.suptitle(
            f'Criterion 2: α_CE Rank Correlation - {status}',
            fontsize=16,
            fontweight='bold'
        )
        
        plt.tight_layout()
        plot_path = self.output_dir / 'criterion_2_plots.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plots to {plot_path}")
        plt.close()

