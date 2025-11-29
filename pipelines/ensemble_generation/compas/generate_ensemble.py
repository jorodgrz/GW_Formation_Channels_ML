#!/usr/bin/env python3
"""
Generate COMPAS Ensemble for Physics-Informed ML Priors

This script generates a systematic grid of COMPAS simulations varying key
astrophysical parameters to quantify epistemic uncertainties in formation
channel inference.

Key Parameters Varied:
    - common_envelope_alpha: α_CE ∈ [0.1, 5.0] (primary parameter)
    - common_envelope_lambda: λ_CE ∈ [0.05, 1.0]
    - kick_magnitude_sigma: Natal kick velocities
    - metallicity: Z ∈ [0.0001, 0.02]
    - mass_transfer_fa: Accretion efficiency
"""

import json
import logging
import os
import subprocess
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_COMPAS_BINARY = REPO_ROOT / "simulators" / "compas" / "src" / "bin" / "COMPAS"
DEFAULT_COMPAS_OUTPUT = REPO_ROOT / "experiments" / "runs" / "compas_ensemble_output"


class COMPASEnsembleGenerator:
    """
    Generates systematic COMPAS ensemble for epistemic uncertainty quantification
    
    This class handles the generation of a large parameter grid and executes
    COMPAS simulations for each parameter combination to build an ensemble
    that captures model uncertainties.
    """
    
    def __init__(
        self,
        compas_binary: str = str(DEFAULT_COMPAS_BINARY),
        output_base: str = str(DEFAULT_COMPAS_OUTPUT),
        n_systems_per_run: int = 100000
    ):
        """
        Initialize the ensemble generator
        
        Args:
            compas_binary: Path to COMPAS executable
            output_base: Base directory for ensemble outputs
            n_systems_per_run: Number of binary systems per simulation
        """
        self.compas_binary = Path(compas_binary).expanduser().resolve()
        self.output_base = Path(output_base).expanduser().resolve()
        self.n_systems_per_run = n_systems_per_run
        
        # Create output directory
        self.output_base.mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata storage
        self.metadata = {
            'creation_time': datetime.now().isoformat(),
            'compas_binary': str(self.compas_binary),
            'n_systems_per_run': n_systems_per_run,
            'runs': []
        }
        
        logger.info(f"Initialized COMPAS Ensemble Generator")
        logger.info(f"Output directory: {self.output_base}")
        logger.info(f"Systems per run: {n_systems_per_run}")
    
    def generate_parameter_grid(
        self,
        n_alpha_points: int = 10,
        use_sparse_grid: bool = False
    ) -> List[Dict]:
        """
        Generate parameter grid for epistemic uncertainty sampling
        
        This creates a systematic grid varying key astrophysical parameters,
        with special focus on α_CE which is the primary driver of Channel I/IV
        degeneracy.
        
        Args:
            n_alpha_points: Number of points in α_CE grid (logarithmically spaced)
            use_sparse_grid: If True, use sparse sampling for faster testing
            
        Returns:
            List of parameter dictionaries for each simulation
        """
        logger.info("Generating parameter grid...")
        
        # CE alpha is the key parameter for Channel I/IV degeneracy
        # Use logarithmic spacing to capture wide range
        alpha_ce_grid = np.logspace(np.log10(0.1), np.log10(5.0), n_alpha_points)
        
        if use_sparse_grid:
            # Sparse grid for testing
            lambda_ce_grid = [0.1, 0.5]
            kick_sigma_grid = [217]  # Standard Hobbs value
            metallicity_grid = [0.0001, 0.0142]  # Low and solar
            fa_grid = [0.5]  # Standard value
        else:
            # Full grid for production
            lambda_ce_grid = [0.05, 0.1, 0.5, 1.0]
            kick_sigma_grid = [150, 217, 300]  # Low, standard, high
            metallicity_grid = [0.0001, 0.001, 0.0142, 0.02]  # Z range
            fa_grid = [0.25, 0.5, 0.75]  # Accretion efficiency
        
        # Generate all combinations
        grid_params = []
        for alpha, lambda_val, kick, Z, fa in product(
            alpha_ce_grid, lambda_ce_grid, kick_sigma_grid, 
            metallicity_grid, fa_grid
        ):
            # Create unique run identifier
            run_id = (
                f"alpha{alpha:.3f}_lambda{lambda_val:.2f}_"
                f"kick{kick}_Z{Z:.5f}_fa{fa:.2f}"
            )
            
            params = {
                'alpha_ce': float(alpha),
                'lambda_ce': float(lambda_val),
                'kick_sigma': float(kick),
                'metallicity': float(Z),
                'mass_transfer_fa': float(fa),
                'run_id': run_id
            }
            grid_params.append(params)
        
        logger.info(f"Generated grid with {len(grid_params)} parameter combinations")
        logger.info(f"  α_CE points: {n_alpha_points}")
        logger.info(f"  λ_CE values: {len(lambda_ce_grid)}")
        logger.info(f"  Kick values: {len(kick_sigma_grid)}")
        logger.info(f"  Metallicities: {len(metallicity_grid)}")
        logger.info(f"  Accretion efficiencies: {len(fa_grid)}")
        
        return grid_params
    
    def run_single_simulation(
        self,
        params: Dict,
        verbose: bool = False
    ) -> Tuple[str, bool]:
        """
        Execute single COMPAS run with specified parameters
        
        Args:
            params: Parameter dictionary for this run
            verbose: If True, print COMPAS output
            
        Returns:
            Tuple of (output_file_path, success_flag)
        """
        run_id = params['run_id']
        run_dir = self.output_base / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Construct COMPAS command
        cmd = [
            str(self.compas_binary),
            '--number-of-systems', str(self.n_systems_per_run),
            '--common-envelope-alpha', str(params['alpha_ce']),
            '--common-envelope-lambda', str(params['lambda_ce']),
            '--kick-magnitude-sigma-CCSN-NS', str(params['kick_sigma']),
            '--kick-magnitude-sigma-CCSN-BH', str(params['kick_sigma']),
            '--metallicity', str(params['metallicity']),
            '--mass-transfer-fa', str(params['mass_transfer_fa']),
            '--output-path', str(run_dir),
            '--output-container', 'COMPAS_Output',
            # Output configuration
            '--logfile-type', 'HDF5',
            '--rlof-printing', 'TRUE',
            '--evolve-unbound-systems', 'FALSE',  # Focus on bound DCOs
            '--detailed-output', 'TRUE',  # Enable detailed CE tracking
            # Population properties
            '--initial-mass-function', 'KROUPA',
            '--semi-major-axis-distribution', 'FLATINLOG',
            # Physics options
            '--common-envelope-allow-main-sequence-survive', 'TRUE',
        ]
        
        logger.info(f"Running: {run_id}")
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            
            if verbose:
                logger.info(result.stdout)
            
            # Check output file exists
            output_file = run_dir / 'COMPAS_Output' / 'COMPAS_Output.h5'
            if not output_file.exists():
                logger.error(f"Output file not found: {output_file}")
                return str(output_file), False
            
            # Record metadata
            self.metadata['runs'].append({
                'run_id': run_id,
                'params': params,
                'output_file': str(output_file),
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"Completed: {run_id}")
            return str(output_file), True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"COMPAS run failed: {run_id}")
            logger.error(f"Error: {e.stderr}")
            
            self.metadata['runs'].append({
                'run_id': run_id,
                'params': params,
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            
            return "", False
    
    def run_ensemble(
        self,
        grid_params: List[Dict],
        max_parallel: int = 1,
        checkpoint_every: int = 10
    ):
        """
        Run entire ensemble of COMPAS simulations
        
        Args:
            grid_params: List of parameter dictionaries
            max_parallel: Maximum parallel runs (1 = sequential)
            checkpoint_every: Save metadata every N runs
        """
        logger.info(f"Starting ensemble generation with {len(grid_params)} runs")
        
        n_success = 0
        n_failed = 0
        
        for i, params in enumerate(grid_params, 1):
            logger.info(f"Progress: {i}/{len(grid_params)}")
            
            output_file, success = self.run_single_simulation(params)
            
            if success:
                n_success += 1
            else:
                n_failed += 1
            
            # Save checkpoint
            if i % checkpoint_every == 0:
                self.save_metadata()
                logger.info(f"Checkpoint: {n_success} success, {n_failed} failed")
        
        # Final save
        self.save_metadata()
        
        logger.info("="*60)
        logger.info("ENSEMBLE GENERATION COMPLETE")
        logger.info(f"Total runs: {len(grid_params)}")
        logger.info(f"Successful: {n_success}")
        logger.info(f"Failed: {n_failed}")
        logger.info(f"Success rate: {n_success/len(grid_params)*100:.1f}%")
        logger.info("="*60)
    
    def save_metadata(self):
        """Save ensemble metadata to JSON file"""
        metadata_file = self.output_base / 'ensemble_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        logger.info(f"Saved metadata: {metadata_file}")
    
    def load_metadata(self) -> Dict:
        """Load existing ensemble metadata"""
        metadata_file = self.output_base / 'ensemble_metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
        return None
    
    def get_successful_runs(self) -> List[Dict]:
        """Get list of successful runs from metadata"""
        return [
            run for run in self.metadata['runs']
            if run['status'] == 'success'
        ]


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate COMPAS ensemble for epistemic uncertainty quantification'
    )
    parser.add_argument(
        '--compas-binary',
        default=str(DEFAULT_COMPAS_BINARY),
        help='Path to COMPAS executable'
    )
    parser.add_argument(
        '--output-dir',
        default=str(DEFAULT_COMPAS_OUTPUT),
        help='Output directory for ensemble'
    )
    parser.add_argument(
        '--n-systems',
        type=int,
        default=100000,
        help='Number of systems per run'
    )
    parser.add_argument(
        '--n-alpha-points',
        type=int,
        default=10,
        help='Number of α_CE grid points'
    )
    parser.add_argument(
        '--sparse',
        action='store_true',
        help='Use sparse grid for testing'
    )
    parser.add_argument(
        '--test-run',
        action='store_true',
        help='Run only first 3 parameter combinations for testing'
    )
    parser.add_argument(
        '--start-index',
        type=int,
        default=0,
        help='Start index within the parameter grid (inclusive)'
    )
    parser.add_argument(
        '--end-index',
        type=int,
        default=None,
        help='End index within the parameter grid (exclusive). Defaults to full grid.'
    )
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = COMPASEnsembleGenerator(
        compas_binary=args.compas_binary,
        output_base=args.output_dir,
        n_systems_per_run=args.n_systems
    )
    
    # Generate parameter grid
    grid_params = generator.generate_parameter_grid(
        n_alpha_points=args.n_alpha_points,
        use_sparse_grid=args.sparse
    )
    
    # For testing, only run first few
    if args.test_run:
        logger.info("TEST MODE: Running only first 3 parameter combinations")
        grid_params = grid_params[:3]
    else:
        start = max(0, args.start_index)
        end = len(grid_params) if args.end_index is None else min(len(grid_params), max(start, args.end_index))
        if start != 0 or end != len(grid_params):
            logger.info(f"Subselecting parameter grid indices [{start}:{end}) out of {len(grid_params)} total")
        grid_params = grid_params[start:end]
    
    if len(grid_params) == 0:
        logger.warning("No parameter combinations selected after slicing; exiting.")
        return
    
    # Run ensemble
    generator.run_ensemble(grid_params)


if __name__ == "__main__":
    main()

