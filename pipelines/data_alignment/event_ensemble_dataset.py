"""
Event-Ensemble Alignment Dataset

Pairs GWTC-4 events with population synthesis ensemble slices so the physics
informed network can learn from both observed and simulated domains.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class SliceEncodingConfig:
    """Configuration describing how ensemble slices are summarized."""

    summary_columns: Sequence[str]
    param_columns: Sequence[str]
    slice_size: int
    input_dim: int


class EnsembleSliceRepository:
    """
    Loads ensemble outputs (COMPAS/COSMIC/…) and returns matched slices.
    """

    def __init__(
        self,
        ensemble_dirs: Dict[str, str],
        code_order: Sequence[str],
        slice_size: int = 256,
        max_systems_per_code: Optional[int] = None,
    ):
        self.slice_size = max(1, slice_size)
        self.code_order = [code.lower() for code in code_order]
        self.max_systems_per_code = max_systems_per_code

        self.summary_columns = [
            "m1_source",
            "m2_source",
            "chirp_mass",
            "mass_ratio",
            "coalescence_time",
        ]
        self.param_columns = [
            "alpha_ce",
            "lambda_ce",
            "kick_sigma",
            "metallicity",
            "mass_transfer_fa",
        ]

        self.code_tables: Dict[str, pd.DataFrame] = {}
        self.column_scales: Dict[Tuple[str, str], float] = {}

        for code, directory in ensemble_dirs.items():
            loader = getattr(self, f"_load_{code.lower()}_ensemble", None)
            if loader is None:
                logger.warning("No loader implemented for code '%s'", code)
                continue
            df = loader(Path(directory))
            if df.empty:
                logger.warning("No systems loaded for %s from %s", code, directory)
                continue
            if self.max_systems_per_code and len(df) > self.max_systems_per_code:
                df = df.sample(
                    n=self.max_systems_per_code,
                    random_state=42,
                    replace=False,
                )
            code_key = code.lower()
            self.code_tables[code_key] = df.reset_index(drop=True)
            for column in self.summary_columns:
                if column in df.columns:
                    scale = float(df[column].std())
                    self.column_scales[(code_key, column)] = scale if scale > 0 else 1.0

        if not self.code_tables:
            logger.warning(
                "No ensemble tables were loaded. Domain encodings will be zeros."
            )

    # ------------------------------------------------------------------
    # Loading utilities
    # ------------------------------------------------------------------
    def _load_compas_ensemble(self, base_dir: Path) -> pd.DataFrame:
        metadata_file = base_dir / "ensemble_metadata.json"
        if not metadata_file.exists():
            logger.warning("COMPAS metadata not found at %s", metadata_file)
            return pd.DataFrame()

        with metadata_file.open("r") as handle:
            metadata = json.load(handle)

        rows: List[pd.DataFrame] = []
        for run in metadata.get("runs", []):
            if run.get("status") != "success":
                continue
            output_path = Path(run["output_file"])
            if not output_path.is_absolute():
                candidate = base_dir / output_path
                if not candidate.exists():
                    candidate_alt = base_dir.parent / output_path
                    if candidate_alt.exists():
                        candidate = candidate_alt
                output_path = candidate
            if not output_path.exists():
                logger.warning("COMPAS output missing: %s", output_path)
                continue
            rows.append(
                self._extract_compas_dcos(
                    output_path,
                    params=run.get("params", {}),
                )
            )

        if not rows:
            return pd.DataFrame()
        df = pd.concat(rows, ignore_index=True)
        df["code"] = "compas"
        return df

    def _extract_compas_dcos(self, file_path: Path, params: Dict) -> pd.DataFrame:
        with h5py.File(file_path, "r") as h5_file:
            dco_group = h5_file.get("BSE_Double_Compact_Objects")
            if dco_group is None:
                return pd.DataFrame()
            mass1 = np.array(dco_group.get("Mass(1)", []), dtype=np.float64)
            mass2 = np.array(dco_group.get("Mass(2)", []), dtype=np.float64)
            t_delay = np.array(
                dco_group.get("Coalescence_Time", np.zeros_like(mass1)),
                dtype=np.float64,
            )
            merges_mask = np.asarray(
                dco_group.get("Merges_Hubble_Time", np.ones_like(mass1)),
                dtype=bool,
            )

        mask = merges_mask.astype(bool)
        mass1 = mass1[mask]
        mass2 = mass2[mask]
        t_delay = t_delay[mask]

        if len(mass1) == 0:
            return pd.DataFrame()

        chirp_mass = self._chirp_mass(mass1, mass2)
        mass_ratio = self._safe_mass_ratio(mass1, mass2)

        df = pd.DataFrame(
            {
                "m1_source": mass1,
                "m2_source": mass2,
                "chirp_mass": chirp_mass,
                "mass_ratio": mass_ratio,
                "coalescence_time": t_delay / 1e3,  # Myr -> Gyr
                "alpha_ce": params.get("alpha_ce", params.get("alpha", 0.0)),
                "lambda_ce": params.get("lambda_ce", params.get("lambda", 0.0)),
                "kick_sigma": params.get("kick_sigma", params.get("sigma", 0.0)),
                "metallicity": params.get("metallicity", 0.0),
                "mass_transfer_fa": params.get("mass_transfer_fa", params.get("fa", 0.0)),
            }
        )
        return df.fillna(0.0)

    def _load_cosmic_ensemble(self, base_dir: Path) -> pd.DataFrame:
        metadata_file = base_dir / "ensemble_metadata.json"
        if not metadata_file.exists():
            logger.warning("COSMIC metadata not found at %s", metadata_file)
            return pd.DataFrame()

        with metadata_file.open("r") as handle:
            metadata = json.load(handle)

        rows: List[pd.DataFrame] = []
        for run in metadata.get("runs", []):
            if run.get("status") != "success":
                continue
            output_path = Path(run["output_file"])
            if not output_path.is_absolute():
                candidate = base_dir / output_path
                if not candidate.exists():
                    candidate_alt = base_dir.parent / output_path
                    if candidate_alt.exists():
                        candidate = candidate_alt
                output_path = candidate
            if not output_path.exists():
                logger.warning("COSMIC output missing: %s", output_path)
                continue
            rows.append(
                self._extract_cosmic_dcos(
                    output_path,
                    params=run.get("params", {}),
                )
            )

        if not rows:
            return pd.DataFrame()
        df = pd.concat(rows, ignore_index=True)
        df["code"] = "cosmic"
        return df

    def _extract_cosmic_dcos(self, file_path: Path, params: Dict) -> pd.DataFrame:
        try:
            dco = pd.read_hdf(file_path, "dcos")
        except Exception as exc:
            logger.warning("Failed to read COSMIC file %s: %s", file_path, exc)
            return pd.DataFrame()

        mass1 = dco.get("mass_1", pd.Series(dtype=float))
        mass2 = dco.get("mass_2", pd.Series(dtype=float))
        if mass1.empty or mass2.empty:
            return pd.DataFrame()

        chirp_mass = self._chirp_mass(mass1.to_numpy(), mass2.to_numpy())
        mass_ratio = self._safe_mass_ratio(mass1.to_numpy(), mass2.to_numpy())

        df = pd.DataFrame(
            {
                "m1_source": mass1.to_numpy(),
                "m2_source": mass2.to_numpy(),
                "chirp_mass": chirp_mass,
                "mass_ratio": mass_ratio,
                "coalescence_time": dco.get("tphys", pd.Series(0.0)).to_numpy()
                / 1e3,  # Myr -> Gyr
                "alpha_ce": params.get("alpha", 0.0),
                "lambda_ce": params.get("lambd", 0.0),
                "kick_sigma": params.get("sigma", 0.0),
                "metallicity": params.get("metallicity", 0.0),
                "mass_transfer_fa": params.get("beta", params.get("fa", 0.0)),
            }
        )
        return df.fillna(0.0)

    @staticmethod
    def _chirp_mass(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
        total = m1 + m2
        eta = (m1 * m2) / np.maximum(total**2, 1e-6)
        return total * np.power(eta, 3.0 / 5.0)

    @staticmethod
    def _safe_mass_ratio(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.minimum(m1 / np.maximum(m2, 1e-6), m2 / np.maximum(m1, 1e-6))
        ratio = np.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0)
        return ratio

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def encode_event(
        self,
        event_features: Dict[str, float],
        config: SliceEncodingConfig,
        code_order: Optional[Sequence[str]] = None,
    ) -> List[torch.Tensor]:
        """
        Return a list of per-code encodings aligned with `code_order`.
        """
        order = [code.lower() for code in (code_order or self.code_order)]
        vectors: List[torch.Tensor] = []

        for code in order:
            df = self.code_tables.get(code)
            if df is None or df.empty:
                vectors.append(torch.zeros(config.input_dim, dtype=torch.float32))
                continue
            encoded = self._encode_slice(df, code, event_features, config)
            vectors.append(torch.from_numpy(encoded))

        return vectors

    def _encode_slice(
        self,
        df: pd.DataFrame,
        code: str,
        event_features: Dict[str, float],
        config: SliceEncodingConfig,
    ) -> np.ndarray:
        if df.empty:
            return np.zeros(config.input_dim, dtype=np.float32)

        scores = np.zeros(len(df), dtype=np.float64)
        for column in self.summary_columns:
            target_value = event_features.get(column)
            if target_value is None or column not in df.columns:
                continue
            scale = self.column_scales.get((code, column), 1.0)
            values = df[column].to_numpy()
            scores += np.square((values - target_value) / (scale + 1e-6))

        ordered_indices = np.argsort(scores)
        slice_indices = ordered_indices[: self.slice_size]
        slice_df = df.iloc[slice_indices]

        stats: List[float] = []
        for column in self.summary_columns:
            if column not in slice_df.columns:
                stats.extend([0.0, 0.0, 0.0, 0.0])
                continue
            column_values = slice_df[column].to_numpy(dtype=np.float32)
            stats.append(float(np.mean(column_values)))
            stats.append(float(np.std(column_values)))
            stats.append(float(np.percentile(column_values, 5)))
            stats.append(float(np.percentile(column_values, 95)))

        for column in self.param_columns:
            if column in slice_df.columns:
                stats.append(float(slice_df[column].iloc[0]))
            else:
                stats.append(0.0)

        vector = np.array(stats, dtype=np.float32)
        if vector.size >= config.input_dim:
            return vector[: config.input_dim]
        padded = np.zeros(config.input_dim, dtype=np.float32)
        padded[: vector.size] = vector
        return padded


class EventEnsembleDataset(Dataset):
    """
    Torch dataset yielding (code_inputs, gw_obs, channel_label, domain_label).
    """

    def __init__(
        self,
        events: Sequence[Dict],
        code_order: Sequence[str],
        ensemble_dirs: Dict[str, str],
        input_dim: int,
        obs_dim: int,
        slice_size: int = 256,
        synthetic_multiplier: float = 1.0,
        include_simulation: bool = True,
        max_systems_per_code: Optional[int] = None,
        seed: int = 42,
    ):
        self.events = list(events)
        self.obs_dim = obs_dim
        self.code_order = [code.lower() for code in code_order]
        self.rng = np.random.default_rng(seed)

        self.slice_repo = EnsembleSliceRepository(
            ensemble_dirs=ensemble_dirs,
            code_order=code_order,
            slice_size=slice_size,
            max_systems_per_code=max_systems_per_code,
        )
        self.encoding_config = SliceEncodingConfig(
            summary_columns=self.slice_repo.summary_columns,
            param_columns=self.slice_repo.param_columns,
            slice_size=slice_size,
            input_dim=input_dim,
        )

        self.samples: List[Dict] = []
        self.samples.extend(self._build_event_samples())

        if include_simulation:
            synthetic_count = max(
                0, int(len(self.samples) * synthetic_multiplier)
            )
            self.samples.extend(self._build_synthetic_samples(synthetic_count))

        if not self.samples:
            raise ValueError("EventEnsembleDataset contains no samples")

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        return self.samples[index]

    @staticmethod
    def collate_fn(batch: Sequence[Dict]):
        if not batch:
            raise ValueError("Empty batch provided to collate_fn")

        n_codes = len(batch[0]["code_inputs"])
        code_tensors = []
        for code_idx in range(n_codes):
            stacked = torch.stack(
                [sample["code_inputs"][code_idx] for sample in batch],
                dim=0,
            )
            code_tensors.append(stacked)

        gw_obs = torch.stack([sample["gw_obs"] for sample in batch], dim=0)
        targets = torch.stack([sample["target"] for sample in batch], dim=0)
        domains = torch.stack([sample["domain"] for sample in batch], dim=0)
        return code_tensors, gw_obs, targets, domains

    # ------------------------------------------------------------------
    # Sample construction
    # ------------------------------------------------------------------
    def _build_event_samples(self) -> List[Dict]:
        samples: List[Dict] = []
        for event in self.events:
            observables = event.get("observables", {})
            features = self._extract_features(observables)
            code_inputs = self.slice_repo.encode_event(
                features, self.encoding_config, self.code_order
            )
            gw_obs = self._prepare_obs_vector(event.get("gw_observations"))
            samples.append(
                {
                    "code_inputs": code_inputs,
                    "gw_obs": gw_obs,
                    "target": torch.tensor(
                        self._assign_channel_label(features), dtype=torch.long
                    ),
                    "domain": torch.tensor(1, dtype=torch.long),
                    "meta": {"name": event.get("name", "unknown")},
                }
            )
        return samples

    def _build_synthetic_samples(self, n_samples: int) -> List[Dict]:
        if n_samples <= 0 or not self.slice_repo.code_tables:
            return []

        # Use first available code as template for sampling parameter space
        base_code = next(iter(self.slice_repo.code_tables))
        base_df = self.slice_repo.code_tables[base_code]
        if base_df.empty:
            return []

        samples: List[Dict] = []
        for _ in range(n_samples):
            seed_row = base_df.iloc[self.rng.integers(0, len(base_df))]
            features = {
                "m1_source": float(seed_row.get("m1_source", 0.0)),
                "m2_source": float(seed_row.get("m2_source", 0.0)),
                "chirp_mass": float(seed_row.get("chirp_mass", 0.0)),
                "mass_ratio": float(seed_row.get("mass_ratio", 1.0)),
                "coalescence_time": float(seed_row.get("coalescence_time", 0.0)),
                "chi_eff": 0.0,
                "chi_p": 0.0,
                "t_delay": float(seed_row.get("coalescence_time", 0.0)),
                "redshift": 0.0,
                "luminosity_distance": 0.0,
            }
            code_inputs = self.slice_repo.encode_event(
                features, self.encoding_config, self.code_order
            )
            gw_obs = self._features_to_obs_tensor(features)
            samples.append(
                {
                    "code_inputs": code_inputs,
                    "gw_obs": gw_obs,
                    "target": torch.tensor(
                        self._assign_channel_label(features), dtype=torch.long
                    ),
                    "domain": torch.tensor(0, dtype=torch.long),
                    "meta": {"name": "synthetic"},
                }
            )
        return samples

    def _extract_features(self, observables: Dict[str, float]) -> Dict[str, float]:
        m1 = observables.get("m1_source") or observables.get("mass_1_source")
        m2 = observables.get("m2_source") or observables.get("mass_2_source")
        if m1 is None or m2 is None:
            m1 = observables.get("mass_1", 0.0)
            m2 = observables.get("mass_2", 0.0)
        features = {
            "m1_source": float(m1 or 0.0),
            "m2_source": float(m2 or 0.0),
            "chirp_mass": float(
                observables.get("chirp_mass_source", observables.get("chirp_mass", 0.0))
            ),
            "mass_ratio": float(
                observables.get(
                    "mass_ratio",
                    self.slice_repo._safe_mass_ratio(
                        np.array([m1 or 1.0]), np.array([m2 or 1.0])
                    )[0],
                )
            ),
            "coalescence_time": float(observables.get("t_delay", 0.0)),
            "chi_eff": float(observables.get("chi_eff", 0.0)),
            "chi_p": float(observables.get("chi_p", 0.0)),
            "t_delay": float(observables.get("t_delay", 0.0)),
            "redshift": float(observables.get("redshift", 0.0)),
            "luminosity_distance": float(observables.get("luminosity_distance", 0.0)),
        }
        return features

    def _prepare_obs_vector(self, gw_obs: Optional[torch.Tensor]) -> torch.Tensor:
        if gw_obs is None:
            return torch.zeros(self.obs_dim, dtype=torch.float32)
        vector = torch.as_tensor(gw_obs, dtype=torch.float32).flatten()
        if vector.numel() >= self.obs_dim:
            return vector[: self.obs_dim]
        padded = torch.zeros(self.obs_dim, dtype=torch.float32)
        padded[: vector.numel()] = vector
        return padded

    def _features_to_obs_tensor(self, features: Dict[str, float]) -> torch.Tensor:
        vector = torch.tensor(
            [
                features.get("m1_source", 0.0),
                features.get("m2_source", 0.0),
                features.get("mass_ratio", 0.0),
                features.get("chirp_mass", 0.0),
                features.get("chi_eff", 0.0),
                features.get("chi_p", 0.0),
                features.get("redshift", 0.0),
                features.get("t_delay", 0.0),
                features.get("luminosity_distance", 0.0),
                features.get("coalescence_time", 0.0),
            ],
            dtype=torch.float32,
        )
        return self._prepare_obs_vector(vector)

    def _assign_channel_label(self, features: Dict[str, float]) -> int:
        chi_eff = features.get("chi_eff", 0.0)
        chi_p = features.get("chi_p", 0.0)
        mass_ratio = features.get("mass_ratio", 1.0)
        t_delay = features.get("t_delay", 5.0)

        if chi_eff > 0.25 or chi_p > 0.5:
            return 2  # spin-aligned / triple-like
        if chi_eff < -0.2:
            return 3  # CE dominated
        if mass_ratio < 0.5:
            return 1  # dynamical channel
        if t_delay > 8.0:
            return 0  # isolated evolution with long delay
        return 0


