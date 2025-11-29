# COMPAS Ensemble Generation Status

**Local Attempt (macOS):** Started November 25, 2025 @ 20:57 PST  
**Current Status:** READY FOR AWS LAUNCH – automation prepared, waiting on EC2 time/credentials  
**Next Actions:** Use `AWS_EC2_RUNBOOK.md` + `scripts/aws_compas_*.sh` to execute slices on AWS and sync back

## Overview

We attempted to run the sparse COMPAS ensemble locally to build multi-code priors for SBI. The job was stopped intentionally on November 26, 2025 after we determined that the wall-clock time (~7–8 days) exceeded what was practical on the laptop. All subsequent production runs will happen on AWS.

## Configuration

### Sparse Grid Parameters
- **Total parameter combinations:** 40
- **Systems per run:** 10,000
- **Output directory:** `experiments/runs/compas_ensemble_sparse/`

### Parameter Ranges
- **α_CE (Common Envelope Efficiency):** 0.1 to 5.0 (10 points, logarithmically spaced)
- **λ_CE (CE Lambda):** [0.1, 0.5] (2 values)
- **Kick σ (Natal Kicks):** 217 km/s (Hobbs standard)
- **Metallicity:** [0.0001, 0.0142] (low and solar)
- **Accretion Efficiency (fa):** 0.5

## Progress

### Local Run Outcome (Nov 26, 2025 @ 06:05 PST)
- **Run 1 of 40:** reached ~36% (≈3,600 systems) before shutdown
- **Reason for stop:** projected total runtime > 7 days on macOS
- **Partial outputs:** moved to `experiments/runs/compas_ensemble_sparse_aborted_20251125/` for reference
- **Metadata:** no successful runs recorded (HDF5 files incomplete)

### Why It Takes So Long
1. **Detailed stellar evolution:** Each binary system requires solving detailed stellar structure equations
2. **Common envelope physics:** Complex CE evolution calculations
3. **Detailed output enabled:** Writing extensive tracking files for each system
4. **Large system count:** 10,000 systems per run to get statistically significant DCO samples

## AWS Cluster Migration Plan

1. **Provision compute** (AWS Batch or EC2 ASG). See `AWS_CLUSTER.md` for details.
2. **Use new CLI flags** `--start-index` / `--end-index` to slice the 40-combination grid across workers.
3. **Write outputs to S3** (recommended path: `s3://astrothesis-compas/ensemble_sparse/`).
4. **Aggregate metadata** by merging worker-specific JSON fragments into a final `ensemble_metadata.json`.
5. **Sync back to local** workstation once the full ensemble completes.
6. **Automation assets:** `scripts/aws_compas_ec2_setup.sh` (bootstrap), `scripts/aws_compas_run_slice.sh` (per-slice runner), `scripts/aws_sync_compas_results.sh` (S3 <-> local), and `pipelines/ensemble_generation/compas/merge_metadata.py` (final metadata merge).

For quick smoke tests on the laptop continue using:
```bash
python -m pipelines.ensemble_generation.compas.generate_ensemble \
    --test-run --sparse --n-systems 100 \
    --output-dir ./experiments/runs/compas_ensemble_sparse
```

## Expected Outputs

### Per-Run Outputs
Each run produces:
- `COMPAS_Output.h5`: Main HDF5 file with DCO parameters (~8-9 MB for 10k systems)
- `Detailed_Output/`: Directory with detailed evolution files
- `Run_Details`: Runtime information

### Ensemble Metadata
- `ensemble_metadata.json`: Tracks all runs, parameters, and completion status
- Updated every 10 runs (checkpoint)

## What the Ensemble Provides

### For Inference Framework
1. **Physics-informed priors:** Learned from COMPAS population synthesis
2. **Epistemic uncertainty quantification:** Coverage of α_CE parameter space
3. **Channel degeneracy resolution:** Data to distinguish Channel I vs Channel IV

### Key Outputs for SBI
- DCO masses (m1, m2)
- Spin parameters (χ_eff, χ_p)
- Time delays
- Final separations
- Formation channels

## Performance Optimization Options

### If Time is Critical
Several options to speed up (can implement after discussing):

1. **Reduce systems per run:** 10,000 → 5,000 (50% time savings)
2. **Disable detailed output:** Remove `--detailed-output` flag (30-40% speedup)
3. **Coarser α_CE grid:** 10 → 6 points (40% fewer runs)
4. **Parallel execution:** Run multiple COMPAS instances simultaneously

### Trade-offs
- Fewer systems = less statistical power
- No detailed output = can't track CE evolution in detail
- Coarser grid = less precise epistemic uncertainty
- Parallel = needs more RAM (~2-3 GB per instance)

## AWS Run Command Template

```bash
python -m pipelines.ensemble_generation.compas.generate_ensemble \
    --sparse \
    --n-systems 10000 \
    --start-index ${START} \
    --end-index ${END} \
    --output-dir /mnt/compas_runs \
    --compas-binary /opt/ASTROTHESIS/simulators/compas/src/bin/COMPAS
```

## Next Steps

### After Sparse Ensemble Completes
1. **Verify outputs:** Check HDF5 files and metadata
2. **Test SBI framework:** Load ensemble data into `sbi_framework.py`
3. **Decide on full grid:** Optionally run full production grid (1,440 combinations)

### Immediate Actions
- Launch EC2 node(s) per `AWS_EC2_RUNBOOK.md` and run the 40 slices via `scripts/aws_compas_run_slice.sh`.
- Monitor uploads under `s3://astrothesis-compas/ensemble_sparse/{runs,logs,metadata}`.
- After syncing to the laptop, merge metadata with `pipelines/ensemble_generation/compas/merge_metadata.py` and update this file with the final location/timestamp.

## Files Created

- `pipelines/ensemble_generation/compas/generate_ensemble.py`: Fixed and working
- `monitor_ensemble.py`: Progress monitoring script
- `ENSEMBLE_STATUS.md`: This file

## Contact

If the run fails or you need to modify:
1. Check terminal output for errors
2. Review `ensemble_metadata.json` for failed runs
3. Can restart from checkpoint if needed

---

**Last Updated:** November 26, 2025, 06:15 PST  
**Next Steps:** Build AWS Batch job definition and launch array jobs

