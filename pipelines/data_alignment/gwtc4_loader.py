"""
GWTC-4 Data Loader

Loads and preprocesses gravitational wave observations from GWTC-4
(Gravitational Wave Transient Catalog 4) for formation channel inference.

Data sources:
    - GWOSC (Gravitational Wave Open Science Center)
    - GWTC-4 posterior samples (HDF5 format)
    - Event metadata and properties
"""

import contextlib
import json
import logging
import ssl
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib import error as url_error
from urllib import request as url_request

import h5py
import numpy as np
import pandas as pd
try:
    import torch
except ImportError:  # pragma: no cover - allow non-torch utilities to run
    torch = None
from astropy import cosmology
from astropy import units as u

logger = logging.getLogger(__name__)

DEFAULT_ZENODO_RECORD = "17014085"
ZENODO_API_TEMPLATE = "https://zenodo.org/api/records/{record_id}"
DEFAULT_EVENT_PREFIX = "IGWN-GWTC4p0-1a206db3d_721"
COMBINED_FILENAME = "GWTC-4_posteriors.h5"

_SSL_CONTEXT = ssl.create_default_context()
_SSL_CONTEXT.check_hostname = False
_SSL_CONTEXT.verify_mode = ssl.CERT_NONE


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
        cosmology_model: str = 'Planck18',
        raw_event_dir: Optional[str] = None,
        auto_download: bool = False,
        zenodo_record_id: str = DEFAULT_ZENODO_RECORD,
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
        self.raw_event_dir = (
            Path(raw_event_dir).expanduser()
            if raw_event_dir
            else self.data_path.parent
        )
        self.auto_download = auto_download
        self.zenodo_record_id = zenodo_record_id
        self._manifest_cache: Optional[Dict[str, Dict]] = None
        
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
        events_list: List[Dict[str, str]] = []

        combined_file = self._combined_file_path()
        if combined_file.exists():
            with h5py.File(combined_file, "r") as handle:
                for event_name in handle.keys():
                    if event_name in {"history", "version"}:
                        continue
                    events_list.append(
                        {
                            "name": event_name,
                            "file": str(combined_file),
                            "type": "BBH",
                        }
                    )

        if not events_list:
            search_dir = (
                self.data_path if self.data_path.is_dir() else self.raw_event_dir
            )
            pattern = "IGWN-GWTC4p0-*combined_PEDataRelease.hdf5"
            for file_path in search_dir.glob(pattern):
                event_name = self._extract_event_name(file_path.name)
                if not event_name:
                    continue
                events_list.append(
                    {
                        "name": event_name,
                        "file": str(file_path),
                        "type": "BBH",
                    }
                )

        logger.info(
            "Found %d GW events in %s",
            len(events_list),
            self.data_path,
        )
        return pd.DataFrame(events_list)

    def _combined_file_path(self) -> Path:
        if self.data_path.is_file():
            return self.data_path
        return self.data_path / COMBINED_FILENAME

    @staticmethod
    def _extract_event_name(filename: str) -> Optional[str]:
        if "GW" not in filename:
            return None

        parts = filename.replace(".hdf5", "").split("-")
        candidates = [part for part in parts if part.startswith("GW")]
        if not candidates:
            return None

        digit_pref = [
            part for part in candidates
            if len(part) > 2 and part[2].isdigit()
        ]
        if digit_pref:
            return digit_pref[0]
        for part in candidates:
            remainder = part[2:]
            if remainder and any(ch.isdigit() for ch in remainder):
                return part
        return candidates[0]
    
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
        posteriors: Dict[str, np.ndarray] = {}
        try:
            source_path, group_name = self._resolve_event_source(event_name)
        except FileNotFoundError as exc:
            logger.error(str(exc))
            raise

        try:
            with h5py.File(source_path, "r") as handle:
                dataset = None
                if group_name and group_name in handle:
                    event_group = handle[group_name]
                    dataset = event_group.get("posterior_samples")
                if dataset is None:
                    dataset = self._load_raw_event_dataset(handle)
                if dataset is None:
                    raise ValueError(
                        f"Posterior samples not found for {event_name} in {source_path}"
                    )
                posteriors = self._structured_dataset_to_dict(dataset, parameters)

        except Exception as exc:
            logger.error("Failed to load posteriors for %s: %s", event_name, exc)
            raise

        logger.info(
            "Loaded %d parameters for %s from %s",
            len(posteriors),
            event_name,
            source_path,
        )
        return posteriors

    def _resolve_event_source(self, event_name: str) -> Tuple[Path, Optional[str]]:
        combined_path = self._combined_file_path()
        if combined_path.exists():
            with h5py.File(combined_path, "r") as handle:
                if event_name in handle:
                    return combined_path, event_name

        raw_file = self._locate_raw_event_file(event_name)
        if raw_file:
            return raw_file, None

        if self.auto_download:
            logger.info("Attempting to download %s from Zenodo", event_name)
            self._download_event_files([event_name])
            raw_file = self._locate_raw_event_file(event_name)
            if raw_file:
                return raw_file, None

        raise FileNotFoundError(
            f"Unable to locate GWTC-4 event '{event_name}'. "
            "Download additional events with download_gwtc4_data."
        )

    def _locate_raw_event_file(self, event_name: str) -> Optional[Path]:
        search_dir = self.raw_event_dir
        pattern = f"*{event_name}*combined_PEDataRelease.hdf5"
        matches = list(search_dir.glob(pattern))
        return matches[0] if matches else None

    def _download_event_files(self, event_names: List[str]) -> None:
        try:
            download_gwtc4_data(
                output_dir=str(self.raw_event_dir),
                record_id=self.zenodo_record_id,
                event_names=event_names,
            )
            build_combined_posterior_file(
                raw_dir=self.raw_event_dir,
                output_file=self._combined_file_path(),
            )
        except Exception as exc:
            logger.warning("Download attempt failed: %s", exc)

    def _load_raw_event_dataset(self, handle: h5py.File) -> Optional[h5py.Dataset]:
        group = choose_waveform_group(handle)
        if group is None:
            return None
        return group.get("posterior_samples")

    def _structured_dataset_to_dict(
        self,
        dataset: h5py.Dataset,
        parameters: Optional[List[str]],
    ) -> Dict[str, np.ndarray]:
        if dataset.dtype.names is None:
            logger.warning("Posterior dataset lacks named fields; returning empty dict")
            return {}

        data = dataset[()]
        available = dataset.dtype.names
        selected = parameters or list(available)
        result: Dict[str, np.ndarray] = {}
        for name in selected:
            if name not in available:
                logger.warning("Parameter %s not present in posterior samples", name)
                continue
            result[name] = np.array(data[name])
        return result


    
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
            derived['m1_source'] = m1
            derived['m2_source'] = m2
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
        
        if torch is None:
            raise ImportError("PyTorch is required to prepare inference tensors.")

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

# ----------------------------------------------------------------------
# Zenodo download helpers
# ----------------------------------------------------------------------
def download_gwtc4_data(
    output_dir: str = "./gwtc4_data",
    record_id: str = DEFAULT_ZENODO_RECORD,
    event_names: Optional[List[str]] = None,
    include_summary: bool = True,
    overwrite: bool = False,
) -> Path:
    """
    Download GWTC-4 posterior files from Zenodo.

    Args:
        output_dir: Directory where downloaded files should be stored.
        record_id: Zenodo record identifier.
        event_names: Optional list of event names (e.g., GW230518_125908) to fetch.
        include_summary: If True, also download the PESummaryTable.
        overwrite: If True, existing files are replaced.
    """
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    manifest = _fetch_zenodo_manifest(record_id)
    tasks: List[Dict] = []

    summary_key = f"{DEFAULT_EVENT_PREFIX}-PESummaryTable.hdf5"
    if include_summary and summary_key in manifest["files"]:
        tasks.append(manifest["files"][summary_key])

    if event_names:
        requested = event_names
    else:
        requested = sorted(manifest["events"].keys())

    for name in requested:
        file_info = manifest["events"].get(name)
        if not file_info:
            logger.warning("Event %s not found in Zenodo record %s", name, record_id)
            continue
        tasks.append(file_info)

    if not tasks:
        logger.info("No files requested for download")
        return output_path

    for file_info in tasks:
        _download_manifest_entry(file_info, output_path, overwrite=overwrite)

    return output_path


def build_combined_posterior_file(
    raw_dir: Path,
    output_file: Optional[Path] = None,
    waveform_priority: Optional[List[str]] = None,
    compression: int = 4,
) -> Path:
    """
    Aggregate individual event files into a single HDF5 for fast loading.
    """
    raw_dir = Path(raw_dir).expanduser()
    output_file = Path(output_file or (raw_dir / COMBINED_FILENAME))

    event_files = sorted(
        raw_dir.rglob("IGWN-GWTC4p0-*combined_PEDataRelease.hdf5")
    )
    if not event_files:
        logger.warning("No GWTC-4 event files found under %s", raw_dir)
        return output_file

    mode = "a" if output_file.exists() else "w"
    added = 0

    with h5py.File(output_file, mode) as combined:
        for event_file in event_files:
            event_name = GWTC4Loader._extract_event_name(event_file.name)
            if not event_name:
                continue
            if event_name in combined:
                continue

            try:
                with h5py.File(event_file, "r") as handle:
                    group = choose_waveform_group(handle, waveform_priority)
                    if group is None or "posterior_samples" not in group:
                        logger.warning("No posterior group found in %s", event_file)
                        continue
                    data = group["posterior_samples"][()]
            except Exception as exc:
                logger.warning("Failed to read %s: %s", event_file, exc)
                continue

            grp = combined.create_group(event_name)
            grp.create_dataset(
                "posterior_samples",
                data=data,
                compression="gzip",
                compression_opts=compression,
            )
            grp.attrs["source_file"] = np.bytes_(event_file.name)
            grp.attrs["waveform_group"] = np.bytes_(group.name)
            added += 1

    if added > 0:
        logger.info("Added %d events to %s", added, output_file)
    return output_file


def choose_waveform_group(
    handle: h5py.File,
    priority: Optional[Sequence[str]] = None,
) -> Optional[h5py.Group]:
    """
    Select the waveform group to use when multiple are available.
    """
    candidates = [
        group
        for group in handle.values()
        if isinstance(group, h5py.Group) and "posterior_samples" in group
    ]
    if not candidates:
        return None

    priority = priority or [
        "C01:Mixed",
        "Mixed",
        "IMRPhenomXPHM",
        "SEOBNRv4PHM",
        "IMRPhenomNSBH",
    ]
    for token in priority:
        for group in candidates:
            if token in group.name:
                return group
    return candidates[0]


def _fetch_zenodo_manifest(record_id: str) -> Dict[str, Dict[str, Dict]]:
    url = ZENODO_API_TEMPLATE.format(record_id=record_id)
    req = url_request.Request(url)
    try:
        with contextlib.closing(url_request.urlopen(req, context=_SSL_CONTEXT)) as resp:
            payload = json.loads(resp.read().decode())
    except url_error.URLError as exc:
        raise RuntimeError(f"Failed to reach Zenodo API: {exc}") from exc

    manifest = {"events": {}, "files": {}}
    for file_info in payload.get("files", []):
        key = file_info.get("key")
        manifest["files"][key] = file_info
        event_name = GWTC4Loader._extract_event_name(key)
        if event_name:
            manifest["events"][event_name] = file_info
    return manifest


def _download_manifest_entry(
    file_info: Dict,
    output_dir: Path,
    overwrite: bool = False,
) -> Path:
    key = file_info["key"]
    dest = output_dir / key
    if dest.exists() and not overwrite:
        logger.info("File %s already exists, skipping download", dest.name)
        return dest

    url = file_info["links"]["self"]
    req = url_request.Request(url)
    logger.info("Downloading %s (%.2f MB)", key, file_info.get("size", 0) / 1e6)

    with contextlib.closing(url_request.urlopen(req, context=_SSL_CONTEXT)) as resp:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as handle:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)

    return dest


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

