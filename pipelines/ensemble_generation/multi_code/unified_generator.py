#!/usr/bin/env python3
"""
Unified Multi-Code Ensemble Generator

This module provides a unified interface for generating ensembles across
multiple population synthesis codes (COMPAS, COSMIC, POSYDON) to quantify
epistemic uncertainty from stellar evolution model systematics.

The key innovation is treating each code as a distinct Bayesian prior:
    p(θ, C) = p(C) p(θ|C)
where C ∈ {COMPAS, COSMIC, POSYDON} is the code identity.
"""

import json
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pipelines.ensemble_generation.compas.generate_ensemble import (
    COMPASEnsembleGenerator,
)
from pipelines.ensemble_generation.cosmic.generate_ensemble import (
    COSMICEnsembleGenerator,
)
from pipelines.ensemble_generation.posydon.generate_ensemble import (
    POSYDONEnsembleGenerator,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS_DIR = REPO_ROOT / "experiments" / "runs"
DEFAULT_MULTI_CODE_OUTPUT = EXPERIMENTS_DIR / "multi_code_ensemble_output"
DEFAULT_COMPAS_BINARY = REPO_ROOT / "simulators" / "compas" / "src" / "bin" / "COMPAS"


class PopSynthCode(Enum):
    """
    Enumeration of supported population synthesis codes
    
    Each code represents a distinct stellar evolution model with
    systematic differences that contribute to epistemic uncertainty.
    """
    COMPAS = "compas"
    COSMIC = "cosmic"
    POSYDON = "posydon"  # To be implemented


class UnifiedEnsembleGenerator:
    """
    Unified interface for multi-code ensemble generation
    
    This class manages ensemble generation across multiple population synthesis
    codes, ensuring consistent parameter mappings and outputs for downstream
    machine learning tasks.
    
    Key Features:
    - Harmonized parameter space across codes
    - Code identity embedding for uncertainty decomposition
    - Consistent formation channel labels
    - Selection effects and detector sensitivity
    """
    
    def __init__(
        self,
        output_base: str = str(DEFAULT_MULTI_CODE_OUTPUT),
        n_systems_per_run: int = 100000,
        codes_to_run: Optional[List[PopSynthCode]] = None,
        compas_binary: Optional[str] = None
    ):
        """
        Initialize unified ensemble generator
        
        Args:
            output_base: Base directory for all ensemble outputs
            n_systems_per_run: Number of systems per run (consistent across codes)
            codes_to_run: List of codes to use (default: all available)
        """
        self.output_base = Path(output_base).expanduser().resolve()
        self.n_systems_per_run = n_systems_per_run
        self.compas_binary = (
            Path(compas_binary).expanduser().resolve()
            if compas_binary
            else DEFAULT_COMPAS_BINARY
        )
        
        # Default to all available codes
        if codes_to_run is None:
            codes_to_run = [PopSynthCode.COMPAS, PopSynthCode.COSMIC]
        
        self.codes_to_run = codes_to_run
        
        # Create code-specific output directories
        self.output_base.mkdir(parents=True, exist_ok=True)
        
        # Initialize code-specific generators
        self.generators = {}
        self._initialize_generators()
        
        # Master metadata
        self.metadata = {
            'creation_time': datetime.now().isoformat(),
            'codes': [code.value for code in codes_to_run],
            'n_systems_per_run': n_systems_per_run,
            'parameter_grid': None,
            'code_runs': {code.value: [] for code in codes_to_run}
        }
        
        logger.info(f"Initialized Unified Multi-Code Ensemble Generator")
        logger.info(f"Codes: {[c.value for c in codes_to_run]}")
        logger.info(f"Output base: {self.output_base}")
    
    def _initialize_generators(self):
        """Initialize code-specific generators"""
        for code in self.codes_to_run:
            code_output = self.output_base / code.value
            
            if code == PopSynthCode.COMPAS:
                self.generators[code] = COMPASEnsembleGenerator(
                    compas_binary=str(self.compas_binary),
                    output_base=str(code_output),
                    n_systems_per_run=self.n_systems_per_run
                )
                logger.info(f"  Initialized COMPAS generator")
                
            elif code == PopSynthCode.COSMIC:
                self.generators[code] = COSMICEnsembleGenerator(
                    output_base=str(code_output),
                    n_systems_per_run=self.n_systems_per_run
                )
                logger.info(f"  Initialized COSMIC generator")
                
            elif code == PopSynthCode.POSYDON:
                self.generators[code] = POSYDONEnsembleGenerator(
                    output_base=str(code_output),
                    n_systems_per_run=self.n_systems_per_run
                )
                logger.info(f"  Initialized POSYDON generator (requires grid setup)")
    
    def generate_harmonized_parameter_grid(
        self,
        n_alpha_points: int = 10,
        use_sparse_grid: bool = False
    ) -> Dict[PopSynthCode, List[Dict]]:
        """
        Generate harmonized parameter grid across all codes
        
        This ensures that all codes sample the same physical parameter space,
        even though their internal parameter names may differ.
        
        Harmonized Parameters:
        - α_CE: Common envelope efficiency [0.1, 5.0]
        - λ_CE: CE lambda parameter [0.05, 1.0]
        - σ_kick: Natal kick velocity dispersion [150, 350] km/s
        - Z: Metallicity [0.0001, 0.02]
        - Accretion efficiency: Code-specific implementations
        
        Args:
            n_alpha_points: Number of α_CE grid points
            use_sparse_grid: Use sparse grid for testing
            
        Returns:
            Dictionary mapping code to its parameter grid
        """
        logger.info("Generating harmonized parameter grid...")
        
        code_grids = {}
        
        for code in self.codes_to_run:
            logger.info(f"  Generating grid for {code.value}...")
            
            # Each generator has its own grid method with code-specific naming
            grid = self.generators[code].generate_parameter_grid(
                n_alpha_points=n_alpha_points,
                use_sparse_grid=use_sparse_grid
            )
            
            # Add code identity to each parameter set
            for param_set in grid:
                param_set['code'] = code.value
                param_set['code_id'] = code.value  # For embeddings
            
            code_grids[code] = grid
            logger.info(f"    Generated {len(grid)} parameter combinations")
        
        # Store in metadata
        self.metadata['parameter_grid'] = {
            'n_alpha_points': n_alpha_points,
            'use_sparse_grid': use_sparse_grid,
            'grid_sizes': {code.value: len(grid) for code, grid in code_grids.items()}
        }
        
        return code_grids
    
    def run_multi_code_ensemble(
        self,
        n_alpha_points: int = 10,
        use_sparse_grid: bool = False,
        test_run: bool = False,
        checkpoint_every: int = 10
    ):
        """
        Run ensemble generation across all codes
        
        This executes the full multi-code ensemble, with each code running
        the same parameter grid to enable direct epistemic uncertainty
        quantification.
        
        Args:
            n_alpha_points: Number of α_CE grid points
            use_sparse_grid: Use sparse grid for testing
            test_run: Run only first 3 combinations per code
            checkpoint_every: Save metadata every N runs
        """
        logger.info("="*60)
        logger.info("STARTING MULTI-CODE ENSEMBLE GENERATION")
        logger.info("="*60)
        
        # Generate harmonized parameter grids
        code_grids = self.generate_harmonized_parameter_grid(
            n_alpha_points=n_alpha_points,
            use_sparse_grid=use_sparse_grid
        )
        
        # For test runs, limit to first 3 combinations
        if test_run:
            logger.info("TEST MODE: Running only first 3 combinations per code")
            code_grids = {
                code: grid[:3] for code, grid in code_grids.items()
            }
        
        # Run each code's ensemble
        for code in self.codes_to_run:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running {code.value.upper()} ensemble")
            logger.info(f"{'='*60}")
            
            grid = code_grids[code]
            generator = self.generators[code]
            
            # Run ensemble for this code
            generator.run_ensemble(grid, checkpoint_every=checkpoint_every)
            
            # Aggregate results into master metadata
            self.metadata['code_runs'][code.value] = generator.get_successful_runs()
            
            logger.info(f"Completed {code.value} ensemble")
        
        # Save master metadata
        self.save_metadata()
        
        # Print summary
        self._print_summary()
    
    def _print_summary(self):
        """Print summary of multi-code ensemble"""
        logger.info("\n" + "="*60)
        logger.info("MULTI-CODE ENSEMBLE SUMMARY")
        logger.info("="*60)
        
        for code in self.codes_to_run:
            runs = self.metadata['code_runs'][code.value]
            n_success = len([r for r in runs if r['status'] == 'success'])
            n_total = len(runs)
            
            logger.info(f"\n{code.value.upper()}:")
            logger.info(f"  Successful runs: {n_success}/{n_total}")
            
            # Calculate total DCOs if available
            total_dcos = 0
            for run in runs:
                if run['status'] == 'success' and 'n_dcos' in run:
                    total_dcos += run['n_dcos']
            
            if total_dcos > 0:
                logger.info(f"  Total DCOs: {total_dcos}")
        
        logger.info("\n" + "="*60)
    
    def save_metadata(self):
        """Save master metadata"""
        metadata_file = self.output_base / 'multi_code_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        logger.info(f"\nSaved master metadata: {metadata_file}")
    
    def load_ensemble_for_training(
        self,
        max_systems_per_code: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load ensemble data from all codes for training
        
        This method loads and harmonizes outputs from all codes into
        a common format suitable for training the physics-informed neural network.
        
        Args:
            max_systems_per_code: Limit systems per code (for memory)
            
        Returns:
            Dictionary mapping code name to DataFrame of DCO properties
        """
        logger.info("Loading multi-code ensemble data...")
        
        ensemble_data = {}
        
        for code in self.codes_to_run:
            logger.info(f"  Loading {code.value} data...")
            
            code_runs = self.metadata['code_runs'][code.value]
            successful_runs = [r for r in code_runs if r['status'] == 'success']
            
            all_dcos = []
            
            for run in successful_runs:
                output_file = run['output_file']
                
                # Load code-specific output format
                if code == PopSynthCode.COMPAS:
                    # COMPAS uses HDF5
                    import h5py
                    with h5py.File(output_file, 'r') as f:
                        # Extract DCO properties
                        # This will need customization based on COMPAS output structure
                        pass
                
                elif code == PopSynthCode.COSMIC:
                    # COSMIC uses HDF5 with pandas
                    dcos = pd.read_hdf(output_file, 'dcos')
                    
                    # Add run metadata
                    dcos['run_id'] = run['run_id']
                    dcos['code'] = code.value
                    dcos['alpha_ce'] = run['params']['alpha']
                    dcos['lambda_ce'] = run['params']['lambd']
                    
                    all_dcos.append(dcos)
            
            if all_dcos:
                code_df = pd.concat(all_dcos, ignore_index=True)
                
                # Limit size if requested
                if max_systems_per_code and len(code_df) > max_systems_per_code:
                    code_df = code_df.sample(n=max_systems_per_code, random_state=42)
                
                ensemble_data[code.value] = code_df
                logger.info(f"    Loaded {len(code_df)} systems from {code.value}")
        
        return ensemble_data
    
    def get_epistemic_uncertainty_statistics(self) -> Dict:
        """
        Calculate epistemic uncertainty statistics across codes
        
        This quantifies how much codes disagree on DCO properties,
        which is a measure of model uncertainty.
        
        Returns:
            Dictionary of epistemic uncertainty metrics
        """
        logger.info("Calculating epistemic uncertainty statistics...")
        
        # Load data
        ensemble_data = self.load_ensemble_for_training()
        
        # Calculate cross-code statistics
        # This is a simplified version - full implementation would compare
        # DCO mass distributions, merger rates, etc. across codes
        
        stats = {
            'n_codes': len(ensemble_data),
            'total_systems': {code: len(df) for code, df in ensemble_data.items()},
            'code_comparison': 'Not yet implemented'
        }
        
        return stats


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate multi-code ensemble for epistemic uncertainty quantification'
    )
    parser.add_argument(
        '--output-dir',
        default=str(DEFAULT_MULTI_CODE_OUTPUT),
        help='Base output directory for all codes'
    )
    parser.add_argument(
        '--n-systems',
        type=int,
        default=100000,
        help='Number of systems per run (consistent across codes)'
    )
    parser.add_argument(
        '--compas-binary',
        default=str(DEFAULT_COMPAS_BINARY),
        help='Path to the COMPAS executable'
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
        help='Run only first 3 parameter combinations per code'
    )
    parser.add_argument(
        '--codes',
        nargs='+',
        default=['compas', 'cosmic'],
        choices=['compas', 'cosmic', 'sevn'],
        help='Which codes to run'
    )
    
    args = parser.parse_args()
    
    # Convert code names to enum
    codes = [PopSynthCode(code) for code in args.codes]
    
    # Initialize unified generator
    generator = UnifiedEnsembleGenerator(
        output_base=args.output_dir,
        n_systems_per_run=args.n_systems,
        codes_to_run=codes,
        compas_binary=args.compas_binary
    )
    
    # Run multi-code ensemble
    generator.run_multi_code_ensemble(
        n_alpha_points=args.n_alpha_points,
        use_sparse_grid=args.sparse,
        test_run=args.test_run
    )


if __name__ == "__main__":
    main()

