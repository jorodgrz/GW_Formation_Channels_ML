#!/usr/bin/env python3
"""
Generate COSMIC Ensemble for Physics-Informed ML Priors

This script generates a systematic grid of COSMIC simulations varying key
astrophysical parameters to quantify epistemic uncertainties in formation
channel inference.

Key Parameters Varied:
    - alpha: α_CE ∈ [0.1, 5.0] (primary parameter)
    - lambd: λ_CE ∈ [0.05, 1.0]
    - sigma: Natal kick velocities
    - metallicity: Z ∈ [0.0001, 0.02]
    - beta: Mass transfer efficiency
"""

import json
import logging
import os
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# COSMIC imports
from cosmic.sample.initialbinarytable import InitialBinaryTable
from cosmic.evolve import Evolve
from cosmic.sample.sampler import independent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_COSMIC_OUTPUT = REPO_ROOT / "experiments" / "runs" / "cosmic_ensemble_output"


class COSMICEnsembleGenerator:
    """
    Generates systematic COSMIC ensemble for epistemic uncertainty quantification
    
    This class handles the generation of a large parameter grid and executes
    COSMIC simulations for each parameter combination to build an ensemble
    that captures model uncertainties.
    """
    
    def __init__(
        self,
        output_base: str = str(DEFAULT_COSMIC_OUTPUT),
        n_systems_per_run: int = 100000
    ):
        """
        Initialize the ensemble generator
        
        Args:
            output_base: Base directory for ensemble outputs
            n_systems_per_run: Number of binary systems per simulation
        """
        self.output_base = Path(output_base).expanduser().resolve()
        self.n_systems_per_run = n_systems_per_run
        
        # Create output directory
        self.output_base.mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata storage
        self.metadata = {
            'creation_time': datetime.now().isoformat(),
            'code': 'COSMIC',
            'version': '3.6.1',
            'n_systems_per_run': n_systems_per_run,
            'runs': []
        }
        
        logger.info(f"Initialized COSMIC Ensemble Generator")
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
            kick_sigma_grid = [265.0]  # Standard COSMIC value (km/s)
            metallicity_grid = [0.0001, 0.0142]  # Low and solar
            beta_grid = [-1.0]  # Standard value (thermal-timescale MT)
        else:
            # Full grid for production
            lambda_ce_grid = [0.05, 0.1, 0.5, 1.0]
            kick_sigma_grid = [150.0, 265.0, 350.0]  # Low, standard, high
            metallicity_grid = [0.0001, 0.001, 0.0142, 0.02]  # Z range
            beta_grid = [-2.0, -1.0, -0.5]  # Conservative, standard, optimistic
        
        # Generate all combinations
        grid_params = []
        for alpha, lambda_val, kick, Z, beta in product(
            alpha_ce_grid, lambda_ce_grid, kick_sigma_grid, 
            metallicity_grid, beta_grid
        ):
            # Create unique run identifier
            run_id = (
                f"alpha{alpha:.3f}_lambda{lambda_val:.2f}_"
                f"kick{kick:.0f}_Z{Z:.5f}_beta{beta:.2f}"
            )
            
            params = {
                'alpha': float(alpha),  # COSMIC uses 'alpha' not 'alpha_ce'
                'lambd': float(lambda_val),  # COSMIC uses 'lambd' not 'lambda_ce'
                'sigma': float(kick),  # Kick velocity dispersion
                'metallicity': float(Z),
                'beta': float(beta),  # Mass transfer efficiency exponent
                'run_id': run_id
            }
            grid_params.append(params)
        
        logger.info(f"Generated grid with {len(grid_params)} parameter combinations")
        logger.info(f"  α_CE points: {n_alpha_points}")
        logger.info(f"  λ_CE values: {len(lambda_ce_grid)}")
        logger.info(f"  Kick values: {len(kick_sigma_grid)}")
        logger.info(f"  Metallicities: {len(metallicity_grid)}")
        logger.info(f"  Beta values: {len(beta_grid)}")
        
        return grid_params
    
    def create_bse_params(self, params: Dict) -> Dict:
        """
        Create BSE (Binary Stellar Evolution) parameters for COSMIC
        
        Args:
            params: Parameter dictionary with user-specified values
            
        Returns:
            Complete BSE parameter dictionary for COSMIC
        """
        # Complete BSE parameter dictionary based on COSMIC v3.6.1 requirements
        # All parameters are required by Evolve.evolve()
        bse_params = {
            # USER-SPECIFIED PARAMETERS
            # Common envelope
            'alpha1': params['alpha'],  # CE efficiency
            'lambdaf': params['lambd'],  # CE lambda
            # Kicks
            'sigma': params['sigma'],  # NS kick dispersion (km/s)
            # Mass transfer
            'beta': params['beta'],  # MT efficiency exponent
            
            # STELLAR WIND PARAMETERS
            'neta': 0.5,  # Reimers mass-loss coefficient
            'bwind': 0.0,  # Binary enhanced mass loss parameter
            'hewind': 1.0,  # Helium star wind factor
            'windflag': 3,  # Wind prescription (3 = Vink+2001)
            
            # COMMON ENVELOPE FLAGS
            'ceflag': 0,  # CE prescription (0 = standard de Kool)
            'cekickflag': 2,  # CE kick flag
            'cemergeflag': 0,  # CE merger flag
            'cehestarflag': 0,  # CE HeHG star flag
            
            # TIDES AND DYNAMICS
            'tflag': 1,  # Tidal circularization
            'ifflag': 0,  # SN fallback prescription (0 = Fryer+2012 rapid)
            'grflag': 1,  # Include gravitational wave radiation
            
            # WHITE DWARF PARAMETERS
            'wdflag': 1,  # WD cooling (1 = standard)
            'wd_mass_lim': 1,  # WD mass limit (0 or 1)
            
            # PAIR INSTABILITY
            'pisn': 45.0,  # PISN lower mass limit (Msun)
            
            # BLACK HOLE PARAMETERS
            'bhflag': 1,  # BH natal kick (1 = fallback-modulated)
            'bhms_coll_flag': 0,  # BH formation from MS collision
            'bhspinflag': 0,  # BH spin evolution
            'bhspinmag': 0.0,  # Initial BH spin magnitude
            'bhsigmafrac': 1.0,  # BH kick fraction
            
            # REMNANT PARAMETERS
            'remnantflag': 4,  # Remnant mass prescription (4 = Fryer+2012 rapid)
            'rtmsflag': 0,  # Treatment of r_{TMS}
            'mxns': 3.0,  # Maximum NS mass (Msun)
            'rembar_massloss': 0.5,  # Remnant baryonic mass loss
            
            # TIMESTEP CONTROL
            'pts1': 0.05,  # Time step control parameter 1
            'pts2': 0.01,  # Time step control parameter 2
            'pts3': 0.02,  # Time step control parameter 3
            
            # SUPERNOVA TYPES
            'ecsn': 2.25,  # Electron capture SN mass limit (Msun)
            'ecsn_mlow': 1.6,  # ECSN lower mass limit (Msun)
            'aic': 1,  # Include accretion induced collapse
            'ussn': 0,  # Ultra-stripped SN flag
            
            # KICK DISTRIBUTION
            'sigmadiv': -20.0,  # Kick dispersion for ECS/AIC
            'polar_kick_angle': 90.0,  # Polar kick angle (degrees)
            'natal_kick_array': [[-100.0, -100.0, -100.0, -100.0, -20.0], 
                                 [-100.0, -100.0, -100.0, -100.0, -20.0]],
            'kickflag': 1,  # Kick mechanism (1-5)
            
            # MASS TRANSFER PARAMETERS
            'qcrit_array': [0.0]*16,  # Critical mass ratio array (16 stellar types)
            'xi': 0.5,  # Wind accretion efficiency
            'acc2': 1.5,  # Bondi-Hoyle accretion exponent
            'epsnov': 0.001,  # Nova efficiency
            'eddfac': 1.0,  # Eddington accretion factor
            'gamma': -2.0,  # Angular momentum prescription
            'don_lim': -1,  # Donor limit (-1 = no limit)
            'acc_lim': -1,  # Accretor limit (-1 = no limit)
            
            # MAGNETIC BRAKING
            'bdecayfac': 1.0,  # Binary magnetic braking decay factor
            'bconst': 3000.0,  # Binary magnetic braking constant
            'ck': 1000.0,  # Circularization timescale
            
            # MASS TRANSFER FLAGS
            'qcflag': 5,  # Critical mass ratio flag
            'eddlimflag': 0,  # Eddington limit flag
            
            # ACCRETION ONTO COMPACT OBJECTS
            'fprimc_array': [2.0/21.0]*16,  # Compact object accretion (16 stellar types)
            
            # REJUVENATION
            'rejuv_fac': 1.0,  # Rejuvenation factor
            'rejuvflag': 0,  # Rejuvenation flag
            'htpmb': 1,  # Hertzsprung gap prescription
            
            # TIDAL ENHANCEMENTS
            'ST_cr': 1,  # Convective/radiative stellar type dependence
            'ST_tide': 0,  # Enhanced tidal dissipation in convective envelopes
            
            # METALLICITY
            'zsun': 0.014,  # Solar metallicity
        }
        
        return bse_params
    
    def run_single_simulation(
        self,
        params: Dict,
        verbose: bool = False
    ) -> Tuple[str, bool]:
        """
        Execute single COSMIC run with specified parameters
        
        Args:
            params: Parameter dictionary for this run
            verbose: If True, print detailed output
            
        Returns:
            Tuple of (output_file_path, success_flag)
        """
        run_id = params['run_id']
        run_dir = self.output_base / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Running: {run_id}")
        
        try:
            # Create BSE parameters
            bse_params = self.create_bse_params(params)
            
            # Generate initial binary population
            # Using COSMIC's independent sampler
            logger.info(f"  Generating {self.n_systems_per_run} initial binaries...")
            
            # Set up initial binary table parameters
            final_kstar1 = [13, 14]  # BH, NS (compact objects)
            final_kstar2 = [13, 14]
            
            # Sample initial binaries
            InitialBinaries, mass_singles, mass_binaries, n_singles, n_binaries = \
                InitialBinaryTable.sampler(
                    'independent',  # Sampling method
                    final_kstar1,
                    final_kstar2,
                    binfrac_model=0.5,  # Binary fraction
                    primary_model='kroupa01',  # IMF
                    ecc_model='sana12',  # Eccentricity distribution
                    porb_model='sana12',  # Period distribution
                    SF_start=13700.0,  # Star formation start (Myr ago)
                    SF_duration=0.0,  # Instantaneous burst
                    met=params['metallicity'],  # Metallicity
                    size=self.n_systems_per_run  # Number of systems
                )
            
            logger.info(f"  Generated {len(InitialBinaries)} initial binaries")
            
            # Evolve the binaries
            logger.info(f"  Evolving binaries...")
            bpp, bcm, initC, kick_info = Evolve.evolve(
                initialbinarytable=InitialBinaries,
                BSEDict=bse_params
            )
            
            logger.info(f"  Evolution complete")
            
            # Filter for double compact objects (DCOs)
            # BCM table contains final states
            # kstar1, kstar2: stellar types (13=NS, 14=BH)
            dcos = bcm[
                (bcm['kstar_1'].isin([13, 14])) & 
                (bcm['kstar_2'].isin([13, 14]))
            ]
            
            logger.info(f"  Found {len(dcos)} DCO systems")
            
            # Save outputs
            output_file = run_dir / 'COSMIC_Output.h5'
            
            # Save all tables to HDF5
            with pd.HDFStore(output_file, 'w') as store:
                store['bpp'] = bpp  # Binary evolution history
                store['bcm'] = bcm  # Binary common envelope
                store['initC'] = initC  # Initial conditions
                store['kick_info'] = kick_info  # Kick information
                store['dcos'] = dcos  # Double compact objects
            
            logger.info(f"  Saved output to {output_file}")
            
            # Record metadata
            self.metadata['runs'].append({
                'run_id': run_id,
                'params': params,
                'output_file': str(output_file),
                'status': 'success',
                'n_systems_total': len(InitialBinaries),
                'n_dcos': len(dcos),
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"Completed: {run_id} ({len(dcos)} DCOs)")
            return str(output_file), True
            
        except Exception as e:
            logger.error(f"COSMIC run failed: {run_id}")
            logger.error(f"Error: {str(e)}")
            
            if verbose:
                import traceback
                traceback.print_exc()
            
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
        checkpoint_every: int = 10
    ):
        """
        Run entire ensemble of COSMIC simulations
        
        Args:
            grid_params: List of parameter dictionaries
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
        description='Generate COSMIC ensemble for epistemic uncertainty quantification'
    )
    parser.add_argument(
        '--output-dir',
        default=str(DEFAULT_COSMIC_OUTPUT),
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
    generator = COSMICEnsembleGenerator(
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

