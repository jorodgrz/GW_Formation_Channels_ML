# AWS Cluster Workflow for COMPAS Ensemble

This guide documents how to execute the multi-code ensemble priors on AWS so the full 40-combination grid finishes within hours instead of days on a laptop.

---

## 1. Architecture Options

| Option | Description | When to use |
| --- | --- | --- |
| **Single large EC2 node** | Launch one compute-optimized instance (e.g., `c7a.16xlarge`) and run 4–8 COMPAS processes in parallel. | Straightforward, manual oversight acceptable. |
| **EC2 Auto Scaling Group** | Use a launch template plus a queue of parameter chunks; each node grabs work from SQS/S3 and reports back. | You want elasticity without Batch. |
| **AWS Batch (job array)** | Submit 40 array jobs, each handling a slice of the grid. Batch handles retries & Spot integration. | Highest automation, best for Spot fleets. |

---

## 2. Base AMI / Bootstrapping

1. **Start from Ubuntu 22.04** (or Amazon Linux 2023) with at least 200 GB gp3 EBS.
2. Install prerequisites:
   ```bash
   sudo apt-get update
   sudo apt-get install -y build-essential cmake git libgsl-dev libhdf5-dev \
       libboost-all-dev libyaml-cpp-dev python3 python3-pip
   ```
3. Install Miniconda (if preferred) and recreate the `gw_channels` environment:
   ```bash
   conda env create -f /path/to/environment.yml  # or pip install -r configs/infrastructure/requirements.txt
   ```
4. Clone this repo (or sync from S3/GitHub) under `/opt/ASTROTHESIS`.
5. Build COMPAS once:
   ```bash
   cd /opt/ASTROTHESIS/COMPAS
   ./scripts/compile.sh  # or equivalent build command
   ```
6. Bake an AMI or keep the bootstrap script handy for User Data.

---

## 3. Preparing Work Units

Each AWS worker should handle a **subset** of the parameter grid. Use the new slicing flags in `generate_ensemble.py`:

```bash
cd /opt/ASTROTHESIS
python -m pipelines.ensemble_generation.compas.generate_ensemble \
    --sparse \
    --n-systems 10000 \
    --start-index ${START} \
    --end-index ${END} \
    --output-dir /mnt/compas_runs/run_${START}_${END}
```

Recommended workflow:

1. Compute the grid length by running `python - <<'PY' ...` locally or on AWS with `--test-run`.
2. For AWS Batch job arrays, set:
   ```bash
   START=$((AWS_BATCH_JOB_ARRAY_INDEX * CHUNK))
   END=$((START + CHUNK))
   ```
3. CHUNK = 1 gives one parameter combo per job (maximum parallelism). CHUNK = 2 or 4 reduces queue size.

---

## 4. Launching Jobs

### Single EC2 Node
1. Launch `c7a.16xlarge` (64 vCPU / 128 GiB). Attach 500 GB gp3 EBS.
2. `tmux` into 8 panes, each running a different `[START:END)` slice.
3. Ensure `ulimit -n` is high (>= 4096) to accommodate HDF5 handles.

### AWS Batch
1. Create a job queue + compute environment (Spot or On-Demand).
2. Package the repo + environment in an ECR image or reference your AMI via Batch custom images.
3. Submit job array:
   ```bash
   aws batch submit-job \
       --job-name compas-ensemble \
       --job-queue gw-channel-queue \
       --job-definition compas-jobdef \
       --array-properties size=40
   ```
4. Your container entrypoint should translate `AWS_BATCH_JOB_ARRAY_INDEX` into `[START:END)` slices.

---

## 5. Storage & Metadata

- Write each run under `/mnt/compas_runs/alpha.../`.
- After completion, sync to S3:
  ```bash
  aws s3 sync /mnt/compas_runs s3://astrothesis-compas/ensemble_sparse/
  ```
- Keep `ensemble_metadata.json` in S3 as the single source of truth. Each worker can append to a DynamoDB table or upload its own metadata fragment that you merge later.

---

## 6. Monitoring & Logging

- **CloudWatch Logs**: send `stdout/stderr` to track COMPAS progress.
- **CloudWatch Metrics**: emit “runs completed” metric every time a job uploads results.
- **Failure retries**: Batch can automatically retry failed slices; for manual setups, wrap each run with `retry` logic (e.g., `for attempt in {1..3}; do ...`).

---

## 7. Cost Planning (On-Demand, Nov 2025 pricing)

| Instance | vCPU | $/hr | 1 run (10 k systems) | 40 runs (sequential) | 40 runs (parallel 8×) |
| --- | --- | --- | --- | --- | --- |
| c7a.8xlarge | 32 | $1.36 | ~1.5 hr | 60 hr / $82 | 7.5 hr / $10.2 per node |
| c7a.16xlarge | 64 | $2.72 | ~0.8 hr | 32 hr / $87 | 4 hr / $10.9 per node |
| c7a.32xlarge | 128 | $5.44 | ~0.45 hr | 18 hr / $98 | 2.3 hr / $12.5 per node |

Spot pricing averages 40–60% cheaper but requires retry logic.

---

## 8. Clean-Up Checklist

- `aws s3 sync` results back to local storage.
- Terminate EC2 instances / disable ASG.
- Delete CloudWatch logs and S3 staging buckets if no longer needed.
- Remove EBS volumes and snapshots.

---

## 9. Quick Reference

```bash
# Example job wrapper (array index provided as $IDX)
START=$((IDX * 1))
END=$((START + 1))

cd /opt/ASTROTHESIS
python -m pipelines.ensemble_generation.compas.generate_ensemble \
    --sparse \
    --n-systems 10000 \
    --start-index ${START} \
    --end-index ${END} \
    --output-dir /mnt/compas_runs
```

Document any completed slices in a shared tracker (S3 JSON, DynamoDB, etc.) so downstream inference modules know which HDF5 files are available. For a concrete EC2 walkthrough (bootstrap → slice runner → S3 sync), consult `AWS_EC2_RUNBOOK.md` and the helper scripts under `scripts/aws_compas_*.sh`.

---

**Last updated:** November 26, 2025

