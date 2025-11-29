#!/usr/bin/env python3
"""
Merge multiple COMPAS ensemble_metadata.json fragments into a single manifest.

Each AWS slice uploads its own metadata file; this utility stitches them
into `experiments/runs/compas_ensemble_sparse/ensemble_metadata.json` so
downstream loaders see the full 40-run ensemble.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge COMPAS ensemble metadata fragments"
    )
    parser.add_argument(
        "--input-dir",
        default="experiments/runs/compas_ensemble_sparse",
        help="Directory that contains slice_* folders (default: %(default)s)",
    )
    parser.add_argument(
        "--pattern",
        default="slice_*/ensemble_metadata.json",
        help="Glob pattern (relative to input-dir) for metadata files",
    )
    parser.add_argument(
        "--output",
        default="experiments/runs/compas_ensemble_sparse/ensemble_metadata.json",
        help="Path for the merged metadata JSON",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and print summary without writing output",
    )
    return parser.parse_args()


def load_metadata_files(files: Iterable[Path]) -> List[Dict]:
    payloads: List[Dict] = []
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            payloads.append(json.load(handle))
    return payloads


def merge_runs(metadata_payloads: Iterable[Dict]) -> List[Dict]:
    seen: Dict[str, Dict] = {}
    for payload in metadata_payloads:
        for run in payload.get("runs", []):
            seen[run["run_id"]] = run
    return [seen[key] for key in sorted(seen.keys())]


def main() -> None:
    args = parse_args()
    root = Path(args.input_dir).expanduser().resolve()
    metadata_paths = sorted(root.glob(args.pattern))

    if not metadata_paths:
        raise SystemExit(
            f"No metadata files matching pattern '{args.pattern}' were found under {root}"
        )

    payloads = load_metadata_files(metadata_paths)
    runs = merge_runs(payloads)

    compas_binary = payloads[-1].get("compas_binary")
    n_systems = payloads[-1].get("n_systems_per_run")

    merged = {
        "creation_time": datetime.utcnow().isoformat() + "Z",
        "source_metadata_files": [str(p) for p in metadata_paths],
        "n_source_files": len(metadata_paths),
        "compas_binary": compas_binary,
        "n_systems_per_run": n_systems,
        "total_runs": len(runs),
        "runs": runs,
    }

    if args.dry_run:
        print(
            f"[dry-run] Would merge {len(metadata_paths)} metadata files containing {len(runs)} runs"
        )
        return

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=2)

    print(
        f"Merged {len(metadata_paths)} metadata files ({len(runs)} runs) into {output_path}"
    )


if __name__ == "__main__":
    main()

