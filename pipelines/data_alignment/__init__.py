"""
Data Loading and Preprocessing Module

Utilities for loading GWTC-4 gravitational wave observations and aligning
them with ensemble population synthesis outputs.
"""

__all__ = [
    "EventEnsembleDataset",
    "GWTC4Loader",
    "download_gwtc4_data",
    "build_combined_posterior_file",
]


def __getattr__(name):
    if name == "EventEnsembleDataset":
        from .event_ensemble_dataset import EventEnsembleDataset as _Dataset

        return _Dataset
    if name == "GWTC4Loader":
        from .gwtc4_loader import GWTC4Loader as _Loader

        return _Loader
    if name == "download_gwtc4_data":
        from .gwtc4_loader import download_gwtc4_data as _download

        return _download
    if name == "build_combined_posterior_file":
        from .gwtc4_loader import build_combined_posterior_file as _build

        return _build
    raise AttributeError(f"module 'pipelines.data_alignment' has no attribute {name}")
