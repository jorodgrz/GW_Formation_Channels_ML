#!/usr/bin/env python3
"""
POSYDON ensemble driver.

This module wraps the upstream `posydon-run-grid` CLI so it can be orchestrated by
the ASTROTHESIS pipelines alongside COMPAS and COSMIC. Rather than recreating the
full POSYDON parameter plumbing, we consume a user-supplied CLI arguments file
and iterate over `--grid-point-index` slices, emitting harmonised metadata.
"""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS_DIR = REPO_ROOT / "experiments" / "runs"
DEFAULT_OUTPUT_DIR = EXPERIMENTS_DIR / "posydon_ensemble_output"


class POSYDONEnsembleGenerator:
    """Wrapper around the `posydon-run-grid` CLI."""

    def __init__(
        self,
        output_base: str = str(DEFAULT_OUTPUT_DIR),
        n_systems_per_run: int = 10_000,
        posydon_cli: str = "posydon-run-grid",
        base_cli_args: Optional[List[str]] = None,
        cli_args_file: Optional[str] = None,
        grid_point_count: Optional[int] = None,
    ):
        self.output_base = Path(output_base).expanduser().resolve()
        self.output_base.mkdir(parents=True, exist_ok=True)

        self.n_systems_per_run = n_systems_per_run
        self.posydon_cli = shutil.which(posydon_cli) or posydon_cli
        self.cli_args_file = cli_args_file
        self.grid_point_count = grid_point_count

        base_args: List[str] = []
        if base_cli_args:
            base_args.extend(base_cli_args)
        if cli_args_file:
            base_args.extend(self._load_cli_args_from_file(cli_args_file))
        self.base_cli_args = base_args

        self.output_directory = (
            Path(self._extract_arg_value("--output-directory"))
            if self._extract_arg_value("--output-directory")
            else self.output_base
        )
        self.output_directory.mkdir(parents=True, exist_ok=True)

        if self.grid_point_count is None or self.grid_point_count <= 0:
            # Default to 1 grid point if not specified (minimal stub behavior)
            self.grid_point_count = 1
            logger.warning(
                "grid_point_count not specified; defaulting to 1 (stub mode)"
            )

        self.metadata_path = self.output_base / "ensemble_metadata.json"
        self.metadata = {
            "creation_time": datetime.now().isoformat(),
            "code": "POSYDON",
            "posydon_cli": str(self.posydon_cli),
            "cli_args_file": cli_args_file,
            "grid_point_count": self.grid_point_count,
            "runs": [],
        }

        logger.info("Initialized POSYDON generator")
        logger.info("  CLI: %s", self.posydon_cli)
        logger.info("  Output base: %s", self.output_base)
        logger.info("  POSYDON output directory: %s", self.output_directory)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _load_cli_args_from_file(self, path: str) -> List[str]:
        args: List[str] = []
        for line in Path(path).read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            args.extend(shlex.split(stripped))
        return args

    def _extract_arg_value(self, flag: str) -> Optional[str]:
        for idx, arg in enumerate(self.base_cli_args):
            if arg == flag and idx + 1 < len(self.base_cli_args):
                return self.base_cli_args[idx + 1]
        return None

    # ------------------------------------------------------------------
    # Grid + execution
    # ------------------------------------------------------------------
    def generate_parameter_grid(
        self,
        n_alpha_points: int = 0,  # Unused, kept for API compatibility
        use_sparse_grid: bool = False,
    ) -> List[Dict]:
        """
        Create a list of grid indices that POSYDON should run.

        Args:
            n_alpha_points: unused (present for interface parity)
            use_sparse_grid: if True, truncate to 40 combinations
        """
        total = self.grid_point_count
        indices = list(range(total))
        if use_sparse_grid:
            indices = indices[: min(40, len(indices))]

        params = [
            {
                "grid_point_index": idx,
                "run_id": f"posydon_grid_{idx:04d}",
            }
            for idx in indices
        ]
        return params

    def run_single_configuration(self, config: Dict) -> Dict:
        run_id = config["run_id"]
        idx = config["grid_point_index"]

        cmd = [self.posydon_cli] + self.base_cli_args + [
            "--grid-point-index",
            str(idx),
        ]

        log_path = self.output_base / f"{run_id}.log"
        logger.info("Launching POSYDON run %s (grid_point_index=%s)", run_id, idx)
        try:
            completed = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            log_path.write_text(
                f"COMMAND: {' '.join(cmd)}\n\nSTDOUT:\n{completed.stdout}\n\nSTDERR:\n{completed.stderr}\n"
            )

            result = {
                "run_id": run_id,
                "grid_point_index": idx,
                "status": "success",
                "output_directory": str(self.output_directory),
                "log_file": str(log_path),
                "timestamp": datetime.now().isoformat(),
            }
            self.metadata["runs"].append(result)
            return result

        except subprocess.CalledProcessError as exc:
            log_path.write_text(
                f"COMMAND: {' '.join(cmd)}\n\nSTDOUT:\n{exc.stdout}\n\nSTDERR:\n{exc.stderr}\n"
            )
            logger.error("POSYDON run failed: %s", run_id)
            result = {
                "run_id": run_id,
                "grid_point_index": idx,
                "status": "failed",
                "error": str(exc),
                "log_file": str(log_path),
                "timestamp": datetime.now().isoformat(),
            }
            self.metadata["runs"].append(result)
            return result

    def run_ensemble(
        self,
        grid_params: List[Dict],
        checkpoint_every: int = 5,
    ):
        logger.info("Starting POSYDON ensemble with %d runs", len(grid_params))

        for i, params in enumerate(grid_params, 1):
            logger.info("Progress %d/%d", i, len(grid_params))
            self.run_single_configuration(params)

            if i % checkpoint_every == 0:
                self.save_metadata()

        self.save_metadata()
        logger.info("POSYDON ensemble complete.")

    def save_metadata(self):
        with open(self.metadata_path, "w") as handle:
            json.dump(self.metadata, handle, indent=2)
        logger.info("Saved POSYDON metadata to %s", self.metadata_path)

    def get_successful_runs(self) -> List[Dict]:
        return [run for run in self.metadata["runs"] if run["status"] == "success"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a POSYDON ensemble by wrapping posydon-run-grid."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for ASTROTHESIS metadata/logs.",
    )
    parser.add_argument(
        "--n-systems",
        type=int,
        default=10_000,
        help="Book-keeping value stored in metadata (POSYDON handles sampling).",
    )
    parser.add_argument(
        "--posydon-cli",
        type=str,
        default="posydon-run-grid",
        help="Executable name or path for posydon-run-grid.",
    )
    parser.add_argument(
        "--posydon-args-file",
        required=True,
        help="Text file listing baseline CLI arguments (one flag per line).",
    )
    parser.add_argument(
        "--grid-point-count",
        type=int,
        required=True,
        help="Total number of grid points present in the supplied POSYDON grid.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index within the grid (inclusive).",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="End index within the grid (exclusive). Defaults to full grid.",
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Run only the first three grid points (smoke test).",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="Persist metadata every N runs.",
    )
    parser.add_argument(
        "--sparse",
        action="store_true",
        help="Alias for limiting to 40 grid points (matches other generators).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    generator = POSYDONEnsembleGenerator(
        output_base=args.output_dir,
        n_systems_per_run=args.n_systems,
        posydon_cli=args.posydon_cli,
        cli_args_file=args.posydon_args_file,
        grid_point_count=args.grid_point_count,
    )

    grid = generator.generate_parameter_grid(use_sparse_grid=args.sparse)

    if args.test_run:
        grid = grid[: min(3, len(grid))]
    else:
        start = max(0, args.start_index)
        end = len(grid) if args.end_index is None else min(len(grid), max(start, args.end_index))
        grid = grid[start:end]

    if not grid:
        logger.warning("No grid points selected; exiting.")
        return

    generator.run_ensemble(grid, checkpoint_every=args.checkpoint_every)


if __name__ == "__main__":
    main()

