#!/usr/bin/env python3
"""
Utility to download the full GWTC-4 posterior catalog with a simple progress bar.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.data_alignment.gwtc4_loader import (  # type: ignore
    build_combined_posterior_file,
    download_gwtc4_data,
    _fetch_zenodo_manifest,
    _download_manifest_entry,
    DEFAULT_EVENT_PREFIX,
)


def _progress_bar(completed: int, total: int, width: int = 40) -> str:
    ratio = completed / total
    filled = math.floor(ratio * width)
    return f"[{'#' * filled}{'.' * (width - filled)}] {completed}/{total} ({ratio:5.1%})"


def download_full_catalog(
    output_dir: Path,
    record_id: str,
    overwrite: bool = False,
) -> None:
    manifest = _fetch_zenodo_manifest(record_id)
    events = sorted(manifest["events"].items())
    total = len(events)
    if total == 0:
        print("No events found in manifest; aborting.")
        return

    output_dir = output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {total} GWTC-4 events in record {record_id}.")
    for idx, (event_name, file_info) in enumerate(events, 1):
        _download_manifest_entry(file_info, output_dir, overwrite=overwrite)
        print(f"{_progress_bar(idx, total)}  {event_name}")

    summary_key = f"{DEFAULT_EVENT_PREFIX}-PESummaryTable.hdf5"
    summary_info = manifest["files"].get(summary_key)
    if summary_info:
        print("Downloading PESummary table...")
        _download_manifest_entry(summary_info, output_dir, overwrite=True)
    else:
        print("Warning: PESummary table not found in manifest.")


def main():
    parser = argparse.ArgumentParser(
        description="Download the full GWTC-4 posterior catalog with progress updates."
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw/gwtc4",
        help="Directory to store individual GWTC-4 event files.",
    )
    parser.add_argument(
        "--record-id",
        default="17014085",
        help="Zenodo record ID for GWTC-4 (default: 17014085).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload files even if they already exist.",
    )
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Skip rebuilding the combined HDF5 file.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    download_full_catalog(output_dir, args.record_id, overwrite=args.overwrite)

    if args.no_aggregate:
        print("Skipping combined HDF5 rebuild (per --no-aggregate).")
        return

    combined_path = Path("data/raw/GWTC-4_posteriors.h5")
    print(f"Building combined catalog at {combined_path} ...")
    build_combined_posterior_file(output_dir, output_file=combined_path)
    print("Done.")


if __name__ == "__main__":
    main()

