# POSYDON Integration Guide

**Status:** Planning, POSYDON adds a third population-synthesis baseline  
**Last Updated:** December 4, 2025

---

## 1. Overview

POSYDON couples detailed MESA stellar evolution tracks with binary population
synthesis samplers and interpolation engines. Integrating it alongside COMPAS
and COSMIC provides a third, modern physics baseline for quantifying epistemic
uncertainty in formation-channel inference.

Key goals:

1. Mirror the CLI exposed by the COMPAS/COSMIC generators so automation scripts
   stay uniform.
2. Preserve the shared HDF5 schema (`/dcos`, `/metadata`, `/channels`) for SBI
   loaders.
3. Support both local smoke tests (100 systems) and AWS-scale runs (10k systems
   per grid point).

---

## 2. Environment Setup

1. Clone POSYDON under `simulators/posydon/posydon` (git submodule or vendored).
2. Build the new conda environment defined in
   `configs/infrastructure/environment-posydon.yml`:

   ```bash
   conda env create -f configs/infrastructure/environment-posydon.yml
   conda activate posydon_env
   ```

   The spec already bundles `mesa_reader>=0.3.5`, `mpi4py`, and the necessary
   compiler toolchain.
3. Install the local POSYDON checkout (once vendored):

   ```bash
   pip install -e simulators/posydon/posydon
   ```

4. Smoke-test the import:

   ```bash
   python -c "import posydon; print(posydon.__version__)"
   ```

---

## 3. Generator Plan

File path: `pipelines/ensemble_generation/posydon/generate_ensemble.py`

Features implemented so far:

- Text-based CLI args file loader (`configs/simulator_templates/POSYDON_CLI_ARGS.example`)
- Wrapper around `posydon-run-grid` with automatic `--grid-point-index` slicing
- Metadata logging + stdout/stderr capture per run

Still TODO:

- `generate_parameter_grid()` returning harmonized α_CE, λ_CE, kick, metallicity,
  and wind-scaling grids identical to COMPAS/COSMIC.
- `run_single_configuration()` that:
  1. Initializes the POSYDON sampler with the requested hyperparameters.
  2. Evolves binaries until DCO formation or termination.
  3. Writes summary statistics and detailed channels to HDF5.
- CLI arguments:
 , `--n-systems`
 , `--output-dir`
 , `--sparse`
 , `--start-index/--end-index`
 , `--test-run`
- Metadata handling identical to COMPAS/COSMIC (JSON shards + merged log).

> **Status:** The module now wraps `posydon-run-grid` via a CLI args template.
> Provide a grid-size and args file to run real simulations. Detailed parameter
> harmonization with COMPAS/COSMIC is still in progress.

### CLI Args File

The generator expects a text file containing baseline arguments (one flag per
line). Start from `configs/simulator_templates/POSYDON_CLI_ARGS.example` and fill
in project-specific paths:

```
--mesa-grid /Users/you/POSYDON/grid_params/defaults/grid_params.ini
--grid-type fixed
--output-directory /Users/you/ASTROTHESIS/experiments/runs/posydon_output
--temporary-directory /Users/you/tmp/posydon_workspace
...
```

The generator automatically appends `--grid-point-index X` before launching
`posydon-run-grid`, so each ASTROTHESIS slice is addressed independently.

### Standalone Usage

```
conda activate posydon_env
python -m pipelines.ensemble_generation.posydon.generate_ensemble \
    --posydon-cli posydon-run-grid \
    --posydon-args-file configs/simulator_templates/POSYDON_CLI_ARGS.example \
    --grid-point-count 40 \
    --output-dir experiments/runs/posydon_ensemble_output \
    --test-run
```

`--grid-point-count` should match the number of combinations encoded in your
POSYDON grid file. Use `--start-index/--end-index` to shard work across AWS.

---

## 4. Validation Checklist

1. **Smoke Tests**
  , [ ] 3 × 100-system runs complete locally in <10 minutes.
  , [ ] HDF5 outputs match schema (validated with `tests/integration`).

2. **Performance Benchmarks**
  , Target 10k-system runtime: <1 hour on c7a.8xlarge (estimate; update once
     measured).
  , Profile key bottlenecks (interpolation vs. binary evolution).

3. **Scientific Consistency**
  , Compare mass, spin, and delay-time distributions against COMPAS/COSMIC for
     overlapping hyperparameters.
  , Document any systematic offsets in `docs/results/posydon_vs_compas.md`.

---

## 5. Integration with Multi-Code Pipeline

After the generator exists:

1. Update `PopSynthCode` enum to enable POSYDON runs (already stubbed).
2. Extend `UnifiedEnsembleGenerator` to instantiate the POSYDON generator.
3. Confirm `UnifiedEnsembleGenerator.load_ensemble_for_training()` can read the
   new files without extra adapters.
4. Add POSYDON to `configs/training/pipeline/default_config.yaml` once data is
   available.

---

## 6. AWS Considerations

- Bake POSYDON into the same AMI as COMPAS if possible to simplify provisioning.
- Ensure the bootstrap scripts under `scripts/aws_compas_*.sh` are duplicated as
  `scripts/aws_posydon_*.sh` for queue-based runs.
- Upload large static grids (if required) to S3 and sync them to `/mnt/cache`
  on worker nodes to avoid repeated downloads.

---

## 7. Next Milestones

1. Finalize environment spec and smoke-test instructions (Due: Dec 8, 2025).
2. Prototype generator stub + CLI (Due: Dec 12, 2025).
3. Run first sparse grid slice locally (Due: Dec 15, 2025).
4. Promote POSYDON to “operational” in README once at least one 10k-system slice
   completes.

Update this document whenever a milestone completes so downstream teams know
when POSYDON-derived priors are available.

