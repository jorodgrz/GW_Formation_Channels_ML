"""
GWTC-4 Data Loader

Loads and preprocesses gravitational wave observations from GWTC-4
(Gravitational Wave Transient Catalog 4) for formation channel inference.

Data sources:
    - GWOSC (Gravitational Wave Open Science Center)
    - GWTC-4 posterior samples (HDF5 format)
    - Event metadata and properties
"""

import h5py
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from astropy import cosmology
from astropy import units as u
import logging

logger = logging.getLogger(__name__)


class GWTC4Loader:
    """
    Loader for GWTC-4 gravitational wave observations
    
    Handles loading posterior samples, extracting relevant parameters,
    and calculating derived quantities for formation channel inference.
    """
    
    def __init__(
        self,
        data_path: str,
        catalog_file: Optional[str] = None,
        cosmology_model: str = 'Planck18'
    ):
        """
        Initialize GWTC-4 loader
        
        Args:
            data_path: Path to GWTC-4 posterior samples directory or HDF5 file
            catalog_file: Optional path to catalog metadata CSV
            cosmology_model: Cosmology model for calculations (Planck18, Planck15, etc.)
        """
        self.data_path = Path(data_path)
        self.catalog_file = catalog_file
        
        # Set cosmology
        if cosmology_model == 'Planck18':
            self.cosmo = cosmology.Planck18
        elif cosmology_model == 'Planck15':
            self.cosmo = cosmology.Planck15
        else:
            raise ValueError(f"Unknown cosmology: {cosmology_model}")
        
        # Event catalog
        self.events = []
        self.catalog_df = None
        
        logger.info(f"Initialized GWTC-4 loader with {cosmology_model} cosmology")
    
    def load_catalog(self, bbh_only: bool = True) -> pd.DataFrame:
        """
        Load GWTC-4 event catalog
        
        Args:
            bbh_only: If True, filter to only Binary Black Hole events
            
        Returns:
            DataFrame with event metadata
        """
        if self.catalog_file is not None and Path(self.catalog_file).exists():
            # Load from provided CSV
            df = pd.read_csv(self.catalog_file)
        else:
            # Create basic catalog from data directory
            df = self.create_catalog_from_files()
        
        # Filter to BBH events if requested
        if bbh_only and 'type' in df.columns:
            df = df[df['type'] == 'BBH']
            logger.info(f"Filtered to {len(df)} BBH events")
        
        self.catalog_df = df
        return df
    
    def create_catalog_from_files(self) -> pd.DataFrame:
        """
        Create catalog from available posterior files
        
        Returns:
            DataFrame with basic event information
        """
        events_list = []
        
        if self.data_path.is_file():
            # Single HDF5 file with multiple events
            with h5py.File(self.data_path, 'r') as f:
                for event_name in f.keys():
                    events_list.append({
                        'name': event_name,
                        'file': str(self.data_path),
                        'type': 'BBH'  # Assume BBH for now
                    })
        else:
            # Directory with individual files
            for file_path in self.data_path.glob('*.h5'):
                event_name = file_path.stem
                events_list.append({
                    'name': event_name,
                    'file': str(file_path),
                    'type': 'BBH'
                })
        
        logger.info(f"Found {len(events_list)} events in {self.data_path}")
        return pd.DataFrame(events_list)
    
    def load_event_posteriors(
        self,
        event_name: str,
        parameters: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Load posterior samples for a specific event
        
        Args:
            event_name: Name of the gravitational wave event
            parameters: List of parameter names to load (None = all)
            
        Returns:
            Dictionary mapping parameter names to posterior samples
        """
        # Find event file
        if self.catalog_df is not None:
            event_row = self.catalog_df[self.catalog_df['name'] == event_name]
            if len(event_row) == 0:
                raise ValueError(f"Event {event_name} not found in catalog")
            file_path = event_row.iloc[0]['file']
        else:
            # Try to find file directly
            file_path = self.data_path / f"{event_name}.h5"
            if not file_path.exists():
                file_path = self.data_path  # Might be single file with all events
        
        # Load posteriors from HDF5
        posteriors = {}
        
        try:
            with h5py.File(file_path, 'r') as f:
                # Navigate to event data
                if event_name in f:
                    event_data = f[event_name]
                else:
                    # Might be in a subgroup
                    event_data = f
                
                # Check for posterior samples group
                if 'posterior_samples' in event_data:
                    samples = event_data['posterior_samples']
                elif 'PublicationSamples' in event_data:
                    samples = event_data['PublicationSamples']
                elif 'IMRPhenomXPHM' in event_data:  # Common waveform approximant
                    samples = event_data['IMRPhenomXPHM']['posterior_samples']
                else:
                    # Try to find any samples group
                    samples = event_data
                
                # Load requested parameters
                if parameters is None:
                    parameters = list(samples.keys())
                
                for param in parameters:
                    if param in samples:
                        posteriors[param] = samples[param][()]
                    else:
                        logger.warning(f"Parameter {param} not found for {event_name}")
        
        except Exception as e:
            logger.error(f"Failed to load posteriors for {event_name}: {e}")
            raise
        
        logger.info(f"Loaded {len(posteriors)} parameters for {event_name}")
        return posteriors
    
    def calculate_derived_parameters(
        self,
        posteriors: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Calculate derived parameters from posteriors
        
        Computes:
            - Source-frame masses (if not present)
            - Effective spin (χ_eff)
            - Precessing spin (χ_p)
            - Mass ratio (q)
            - Chirp mass (M_c)
            - Delay time (t_delay)
        
        Args:
            posteriors: Dictionary of posterior samples
            
        Returns:
            Updated posteriors with derived parameters
        """
        derived = posteriors.copy()
        
        # Extract basic parameters
        if 'mass_1_source' in posteriors:
            m1 = posteriors['mass_1_source']
            m2 = posteriors['mass_2_source']
        elif 'm1_source' in posteriors:
            m1 = posteriors['m1_source']
            m2 = posteriors['m2_source']
        elif 'mass_1' in posteriors and 'redshift' in posteriors:
            # Convert detector frame to source frame
            z = posteriors['redshift']
            m1 = posteriors['mass_1'] / (1 + z)
            m2 = posteriors['mass_2'] / (1 + z)
            derived['m1_source'] = m1
            derived['m2_source'] = m2
        else:
            logger.warning("Could not find mass parameters")
            return derived
        
        # Mass ratio (convention: q ≤ 1)
        if 'mass_ratio' not in derived:
            derived['mass_ratio'] = np.minimum(m1/m2, m2/m1)
        
        # Chirp mass
        if 'chirp_mass_source' not in derived:
            M_total = m1 + m2
            eta = (m1 * m2) / (M_total ** 2)
            derived['chirp_mass_source'] = M_total * (eta ** 0.6)
        
        # Effective spin χ_eff (if spins available)
        if 'chi_eff' not in derived:
            if 'spin_1z' in posteriors and 'spin_2z' in posteriors:
                s1z = posteriors['spin_1z']
                s2z = posteriors['spin_2z']
                M_total = m1 + m2
                derived['chi_eff'] = (m1 * s1z + m2 * s2z) / M_total
            else:
                logger.warning("Spin parameters not found, chi_eff not calculated")
        
        # Precessing spin χ_p
        if 'chi_p' not in derived:
            if 'a_1' in posteriors and 'a_2' in posteriors:
                a1 = posteriors['a_1']
                a2 = posteriors['a_2']
                tilt1 = posteriors.get('tilt_1', np.zeros_like(a1))
                tilt2 = posteriors.get('tilt_2', np.zeros_like(a2))
                
                s1_perp = a1 * np.sin(tilt1)
                s2_perp = a2 * np.sin(tilt2)
                
                q = np.minimum(m1/m2, m2/m1)
                B1 = 2 + 3*q / 2
                B2 = 2 + 3/(2*q)
                
                derived['chi_p'] = np.maximum(
                    B1 * s1_perp,
                    B2 * s2_perp
                )
        
        # Delay time from redshift
        if 'redshift' in posteriors:
            z = posteriors['redshift']
            t_lookback = self.cosmo.lookback_time(z).to(u.Gyr).value
            t_universe = self.cosmo.age(0).to(u.Gyr).value
            derived['t_delay'] = t_universe - t_lookback
        
        return derived
    
    def extract_observables(
        self,
        posteriors: Dict[str, np.ndarray],
        summary_stat: str = 'median'
    ) -> Dict[str, float]:
        """
        Extract point estimates and uncertainties from posteriors
        
        Args:
            posteriors: Posterior samples
            summary_stat: Summary statistic ('median', 'mean', 'mode')
            
        Returns:
            Dictionary with point estimates and uncertainties
        """
        observables = {}
        
        for param_name, samples in posteriors.items():
            # Point estimate
            if summary_stat == 'median':
                point_est = np.median(samples)
            elif summary_stat == 'mean':
                point_est = np.mean(samples)
            elif summary_stat == 'mode':
                # Use KDE to find mode
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(samples)
                x_grid = np.linspace(samples.min(), samples.max(), 1000)
                point_est = x_grid[np.argmax(kde(x_grid))]
            else:
                point_est = np.median(samples)
            
            # Uncertainty (standard deviation)
            uncertainty = np.std(samples)
            
            # 90% credible interval
            lower_90 = np.percentile(samples, 5)
            upper_90 = np.percentile(samples, 95)
            
            observables[param_name] = point_est
            observables[f'{param_name}_std'] = uncertainty
            observables[f'{param_name}_lower'] = lower_90
            observables[f'{param_name}_upper'] = upper_90
        
        return observables
    
    def prepare_for_inference(
        self,
        event_name: str,
        n_posterior_samples: int = 1000
    ) -> Dict:
        """
        Prepare event data for formation channel inference
        
        Args:
            event_name: Name of GW event
            n_posterior_samples: Number of posterior samples to use
            
        Returns:
            Dictionary with formatted data for model input
        """
        # Load posteriors
        posteriors = self.load_event_posteriors(event_name)
        
        # Calculate derived parameters
        posteriors = self.calculate_derived_parameters(posteriors)
        
        # Extract observables
        observables = self.extract_observables(posteriors)
        
        # Format for model input
        # Select key parameters for GW observations
        gw_params = [
            'm1_source', 'm2_source', 'mass_ratio', 'chirp_mass_source',
            'chi_eff', 'chi_p', 'redshift', 't_delay',
            'luminosity_distance', 'a_final'
        ]
        
        gw_obs_vector = []
        for param in gw_params:
            if param in observables:
                gw_obs_vector.append(observables[param])
            else:
                gw_obs_vector.append(0.0)  # Placeholder for missing params
        
        # Estimate observational uncertainty (mean posterior std)
        obs_uncertainty = np.mean([
            observables.get('m1_source_std', 0),
            observables.get('m2_source_std', 0),
            observables.get('chi_eff_std', 0)
        ])
        
        # Subsample posteriors for efficiency
        n_samples = min(n_posterior_samples, len(posteriors['m1_source']))
        indices = np.random.choice(
            len(posteriors['m1_source']),
            size=n_samples,
            replace=False
        )
        
        posterior_samples = {
            param: samples[indices]
            for param, samples in posteriors.items()
        }
        
        return {
            'name': event_name,
            'gw_observations': torch.tensor(gw_obs_vector, dtype=torch.float32),
            'observables': observables,
            'observational_uncertainty': obs_uncertainty,
            'posterior_samples': posterior_samples,
            'posteriors': posteriors  # Full posteriors
        }
    
    def load_all_events(
        self,
        max_events: Optional[int] = None,
        bbh_only: bool = True
    ) -> List[Dict]:
        """
        Load and prepare all events in catalog
        
        Args:
            max_events: Maximum number of events to load (None = all)
            bbh_only: Filter to BBH events only
            
        Returns:
            List of prepared event dictionaries
        """
        # Load catalog
        if self.catalog_df is None:
            self.load_catalog(bbh_only=bbh_only)
        
        # Limit number of events
        event_names = self.catalog_df['name'].tolist()
        if max_events is not None:
            event_names = event_names[:max_events]
        
        # Load all events
        events = []
        for i, event_name in enumerate(event_names, 1):
            logger.info(f"Loading event {i}/{len(event_names)}: {event_name}")
            
            try:
                event_data = self.prepare_for_inference(event_name)
                events.append(event_data)
            except Exception as e:
                logger.error(f"Failed to load {event_name}: {e}")
                continue
        
        logger.info(f"Successfully loaded {len(events)} events")
        return events


def download_gwtc4_data(
    output_dir: str = './gwtc4_data',
    source: str = 'zenodo'
) -> Path:
    """
    Download GWTC-4 posterior samples from GWOSC/Zenodo
    
    Args:
        output_dir: Directory to save downloaded data
        source: Data source ('zenodo', 'gwosc')
        
    Returns:
        Path to downloaded data directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading GWTC-4 data to {output_path}")
    
    if source == 'zenodo':
        # GWTC-4 Zenodo record
        zenodo_url = "https://zenodo.org/record/8177023/files/GWTC-4_posteriors.h5"
        
        logger.info("Note: Please download GWTC-4 data manually from:")
        logger.info(f"  {zenodo_url}")
        logger.info(f"Save to: {output_path / 'GWTC-4_posteriors.h5'}")
        
    elif source == 'gwosc':
        logger.info("Download from GWOSC: https://www.gw-openscience.org/GWTC-4/")
    
    return output_path


def create_synthetic_gwtc4_for_testing(
    n_events: int = 20,
    output_file: str = './test_gwtc4.h5'
) -> str:
    """
    Create synthetic GWTC-4-like data for testing
    
    Args:
        n_events: Number of synthetic events to create
        output_file: Output HDF5 file path
        
    Returns:
        Path to created file
    """
    logger.info(f"Creating {n_events} synthetic GW events")
    
    with h5py.File(output_file, 'w') as f:
        for i in range(n_events):
            event_name = f'GW_TEST_{i:03d}'
            event_group = f.create_group(event_name)
            samples_group = event_group.create_group('posterior_samples')
            
            # Generate synthetic posteriors
            n_samples = 5000
            
            # Masses (log-normal distribution)
            m1 = np.random.lognormal(np.log(30), 0.5, n_samples)
            m2 = m1 * np.random.uniform(0.3, 1.0, n_samples)
            
            # Ensure m1 > m2
            m1, m2 = np.maximum(m1, m2), np.minimum(m1, m2)
            
            # Spins
            chi_eff = np.random.normal(0, 0.2, n_samples)
            chi_p = np.abs(np.random.normal(0.3, 0.2, n_samples))
            
            # Redshift
            z = np.random.gamma(2, 0.2, n_samples)
            
            # Distance
            d_L = np.random.gamma(5, 200, n_samples)
            
            # Save to HDF5
            samples_group.create_dataset('m1_source', data=m1)
            samples_group.create_dataset('m2_source', data=m2)
            samples_group.create_dataset('chi_eff', data=chi_eff)
            samples_group.create_dataset('chi_p', data=chi_p)
            samples_group.create_dataset('redshift', data=z)
            samples_group.create_dataset('luminosity_distance', data=d_L)
    
    logger.info(f"Created synthetic data: {output_file}")
    return output_file

