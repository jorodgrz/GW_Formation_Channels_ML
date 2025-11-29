## AWS EC2 Runbook – COMPAS Sparse Grid (40 combinations)

**Goal:** finish the 40×10 k-system sparse COMPAS ensemble, upload every artifact to `s3://astrothesis-compas/ensemble_sparse`, and sync it back to the laptop.

This runbook covers the single-node EC2 workflow described in `AWS_CLUSTER.md`. It assumes an Ubuntu 22.04 AMI and IAM permissions for EC2, S3, and CloudWatch Logs.

---

### 0. Prerequisites

- **S3 bucket:** `astrothesis-compas` (or override `S3_URI` env var). Create folders: `runs/`, `logs/`, `metadata/`.
- **IAM role/user:** needs `AmazonS3FullAccess` (or scoped access to the bucket) and `CloudWatchAgentServerPolicy` if streaming logs.
- **Key pair + security group:** allow SSH (`22/tcp`) from trusted IP, optional CloudWatch agent port.
- **Instance profile storage:** gp3 500 GB (minimum) mounted as root (`/`) plus optional NVMe scratch.

Recommended instance matrix:

| Instance | vCPU | RAM | Notes |
| --- | --- | --- | --- |
| `c7a.16xlarge` | 64 | 128 GiB | Run 8 slices concurrently (≈4 h wall-clock) |
| `c7a.32xlarge` | 128 | 256 GiB | Run 12–16 slices concurrently (≈2 h) |
| `m7a.16xlarge` | 64 | 256 GiB | Use if you need extra RAM for detailed output |

---

### 1. Launch the node and bootstrap

1. **Copy the repo (if not pulling from GitHub):**
   ```bash
   tar czf ASTROTHESIS.tar.gz ASTROTHESIS
   scp ASTROTHESIS.tar.gz ubuntu@ec2-host:~
   ```

2. **Launch EC2** (example CLI):
   ```bash
   aws ec2 run-instances \
     --image-id ami-xxxxxxxx \
     --instance-type c7a.16xlarge \
     --iam-instance-profile Name=astrothesis-compas \
     --key-name astro-key \
     --security-group-ids sg-xxxx \
     --subnet-id subnet-xxxx \
     --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":500,"VolumeType":"gp3"}}]'
   ```

3. **SSH in** and unpack the repo (if copied manually):
   ```bash
   ssh ubuntu@ec2-host
   sudo mkdir -p /opt
   sudo chown ubuntu:ubuntu /opt
   tar xzf ASTROTHESIS.tar.gz -C /opt
   ```

4. **Run the bootstrap helper** (installs apt deps, Miniconda, env, and builds COMPAS):
   ```bash
   cd /opt/ASTROTHESIS
   bash scripts/aws_compas_ec2_setup.sh
   ```

   Overrides (optional):
   ```bash
   REPO_URL=git@github.com:... \
   GIT_REF=prod \
   RUN_ROOT=/data/compas_runs \
   bash scripts/aws_compas_ec2_setup.sh
   ```

5. **Verify**:
   ```bash
   source /opt/miniconda3/etc/profile.d/conda.sh
   conda activate gw_channels
   python -m pipelines.ensemble_generation.compas.generate_ensemble --test-run --sparse --n-systems 10
   ```

---

### 2. Run the 40 sparse-grid slices

Each slice corresponds to one index of the sparse grid. The helper script automatically:
1. Runs `generate_ensemble.py` for the requested `[START, END)` range.
2. Writes outputs to `/mnt/compas_runs/slice_<idx>`.
3. Uploads the slice (outputs + metadata + logs) to S3 under `runs/slice_<idx>/`.

**Single slice example (index 17):**
```bash
START_INDEX=17 \
S3_URI=s3://astrothesis-compas/ensemble_sparse \
OMP_THREADS=6 \
bash scripts/aws_compas_run_slice.sh
```

**Parallel launch (8 at a time on c7a.16xlarge):**
```bash
seq 0 39 | xargs -I{} -P 8 bash -c '
  export START_INDEX={}
  export S3_URI=s3://astrothesis-compas/ensemble_sparse
  export OMP_THREADS=6
  bash scripts/aws_compas_run_slice.sh
'
```

Notes:
- `CHUNK_SIZE` can be set to >1 to let a job handle consecutive indices.
- `aws_compas_run_slice.sh` raises `ulimit -n` and writes logs to `slice_<idx>.log`.
- To resume a failed slice, just rerun with the same `START_INDEX`; COMPAS will overwrite the slice directory.

---

### 3. Parameter index map

| Idx | α_CE | λ_CE | Z | run_id |
| --- | --- | --- | --- | --- |
| 0 | 0.100 | 0.10 | 0.00010 | alpha0.100_lambda0.10_kick217_Z0.00010_fa0.50 |
| 1 | 0.100 | 0.10 | 0.01420 | alpha0.100_lambda0.10_kick217_Z0.01420_fa0.50 |
| 2 | 0.100 | 0.50 | 0.00010 | alpha0.100_lambda0.50_kick217_Z0.00010_fa0.50 |
| 3 | 0.100 | 0.50 | 0.01420 | alpha0.100_lambda0.50_kick217_Z0.01420_fa0.50 |
| 4 | 0.154 | 0.10 | 0.00010 | alpha0.154_lambda0.10_kick217_Z0.00010_fa0.50 |
| 5 | 0.154 | 0.10 | 0.01420 | alpha0.154_lambda0.10_kick217_Z0.01420_fa0.50 |
| 6 | 0.154 | 0.50 | 0.00010 | alpha0.154_lambda0.50_kick217_Z0.00010_fa0.50 |
| 7 | 0.154 | 0.50 | 0.01420 | alpha0.154_lambda0.50_kick217_Z0.01420_fa0.50 |
| 8 | 0.239 | 0.10 | 0.00010 | alpha0.239_lambda0.10_kick217_Z0.00010_fa0.50 |
| 9 | 0.239 | 0.10 | 0.01420 | alpha0.239_lambda0.10_kick217_Z0.01420_fa0.50 |
| 10 | 0.239 | 0.50 | 0.00010 | alpha0.239_lambda0.50_kick217_Z0.00010_fa0.50 |
| 11 | 0.239 | 0.50 | 0.01420 | alpha0.239_lambda0.50_kick217_Z0.01420_fa0.50 |
| 12 | 0.368 | 0.10 | 0.00010 | alpha0.368_lambda0.10_kick217_Z0.00010_fa0.50 |
| 13 | 0.368 | 0.10 | 0.01420 | alpha0.368_lambda0.10_kick217_Z0.01420_fa0.50 |
| 14 | 0.368 | 0.50 | 0.00010 | alpha0.368_lambda0.50_kick217_Z0.00010_fa0.50 |
| 15 | 0.368 | 0.50 | 0.01420 | alpha0.368_lambda0.50_kick217_Z0.01420_fa0.50 |
| 16 | 0.569 | 0.10 | 0.00010 | alpha0.569_lambda0.10_kick217_Z0.00010_fa0.50 |
| 17 | 0.569 | 0.10 | 0.01420 | alpha0.569_lambda0.10_kick217_Z0.01420_fa0.50 |
| 18 | 0.569 | 0.50 | 0.00010 | alpha0.569_lambda0.50_kick217_Z0.00010_fa0.50 |
| 19 | 0.569 | 0.50 | 0.01420 | alpha0.569_lambda0.50_kick217_Z0.01420_fa0.50 |
| 20 | 0.879 | 0.10 | 0.00010 | alpha0.879_lambda0.10_kick217_Z0.00010_fa0.50 |
| 21 | 0.879 | 0.10 | 0.01420 | alpha0.879_lambda0.10_kick217_Z0.01420_fa0.50 |
| 22 | 0.879 | 0.50 | 0.00010 | alpha0.879_lambda0.50_kick217_Z0.00010_fa0.50 |
| 23 | 0.879 | 0.50 | 0.01420 | alpha0.879_lambda0.50_kick217_Z0.01420_fa0.50 |
| 24 | 1.357 | 0.10 | 0.00010 | alpha1.357_lambda0.10_kick217_Z0.00010_fa0.50 |
| 25 | 1.357 | 0.10 | 0.01420 | alpha1.357_lambda0.10_kick217_Z0.01420_fa0.50 |
| 26 | 1.357 | 0.50 | 0.00010 | alpha1.357_lambda0.50_kick217_Z0.00010_fa0.50 |
| 27 | 1.357 | 0.50 | 0.01420 | alpha1.357_lambda0.50_kick217_Z0.01420_fa0.50 |
| 28 | 2.096 | 0.10 | 0.00010 | alpha2.096_lambda0.10_kick217_Z0.00010_fa0.50 |
| 29 | 2.096 | 0.10 | 0.01420 | alpha2.096_lambda0.10_kick217_Z0.01420_fa0.50 |
| 30 | 2.096 | 0.50 | 0.00010 | alpha2.096_lambda0.50_kick217_Z0.00010_fa0.50 |
| 31 | 2.096 | 0.50 | 0.01420 | alpha2.096_lambda0.50_kick217_Z0.01420_fa0.50 |
| 32 | 3.237 | 0.10 | 0.00010 | alpha3.237_lambda0.10_kick217_Z0.00010_fa0.50 |
| 33 | 3.237 | 0.10 | 0.01420 | alpha3.237_lambda0.10_kick217_Z0.01420_fa0.50 |
| 34 | 3.237 | 0.50 | 0.00010 | alpha3.237_lambda0.50_kick217_Z0.00010_fa0.50 |
| 35 | 3.237 | 0.50 | 0.01420 | alpha3.237_lambda0.50_kick217_Z0.01420_fa0.50 |
| 36 | 5.000 | 0.10 | 0.00010 | alpha5.000_lambda0.10_kick217_Z0.00010_fa0.50 |
| 37 | 5.000 | 0.10 | 0.01420 | alpha5.000_lambda0.10_kick217_Z0.01420_fa0.50 |
| 38 | 5.000 | 0.50 | 0.00010 | alpha5.000_lambda0.50_kick217_Z0.00010_fa0.50 |
| 39 | 5.000 | 0.50 | 0.01420 | alpha5.000_lambda0.50_kick217_Z0.01420_fa0.50 |

---

### 4. Upload verification

After each slice finishes:

```bash
aws s3 ls s3://astrothesis-compas/ensemble_sparse/runs/slice_17-17/
aws s3 cp s3://astrothesis-compas/ensemble_sparse/metadata/slice_17-17.json -
```

Expected size per slice:
- `COMPAS_Output/COMPAS_Output.h5` ≈ 8–10 MB (10 k systems, HDF5 only).
- `Detailed_Output/*` ≈ 150–200 MB (depends on CE events).

Use CloudWatch logs to watch `slice_<idx>.log` streams if you forward them.

---

### 5. Sync back to the laptop

On the MacBook (with AWS CLI configured):
```bash
cd /Users/josephrodriguez/ASTROTHESIS
MODE=down \
LOCAL_DIR=experiments/runs/compas_ensemble_sparse \
bash scripts/aws_sync_compas_results.sh
```

Then merge metadata fragments:
```bash
python -m pipelines.ensemble_generation.compas.merge_metadata \
  --input-dir experiments/runs/compas_ensemble_sparse \
  --pattern 'slice_*/ensemble_metadata.json' \
  --output experiments/runs/compas_ensemble_sparse/ensemble_metadata.json
```

Validate counts:
```bash
python - <<'PY'
from pathlib import Path
import h5py, json

root = Path("experiments/runs/compas_ensemble_sparse")
with open(root / "ensemble_metadata.json") as f:
    meta = json.load(f)
print("Runs:", len(meta["runs"]))

sample = next(root.glob("slice_*/alpha*/COMPAS_Output/COMPAS_Output.h5"))
with h5py.File(sample, "r") as h5:
    print("Systems in sample:", len(h5["BSE_System_Parameters"]["M1"]))
PY
```

---

### 6. Troubleshooting & tips

- **AWS CLI auth failures:** run `aws sts get-caller-identity` on EC2 to ensure the role/user is attached.
- **`COMPAS_Output.h5` missing:** check `slice_<idx>.log` for compile/runtime errors; rerun the slice.
- **File descriptor limits:** `aws_compas_run_slice.sh` sets `ulimit -n 65535`, but if system-wide limits block it, edit `/etc/security/limits.conf`.
- **Storage pressure:** expect 40×(≈250 MB) ≈ 10 GB per slice with detailed output; keep ~50 GB buffer.
- **Performance:** `OMP_THREADS=6` per slice is a good balance on c7a.16xlarge (8 concurrent slices).
- **Cleaning up:** after verifying uploads, `aws ec2 terminate-instances --instance-ids i-xxxx`.

---

**Artifacts staged by the scripts**

- `runs/slice_<idx>/alpha.../COMPAS_Output/...` – raw outputs.
- `runs/slice_<idx>/ensemble_metadata.json` – per-slice metadata.
- `logs/slice_<idx>.log` – stdout/stderr for traceability.
- `metadata/slice_<idx>.json` – compact JSON summary for dashboards.

Once all slices finish, you will have 40 metadata summaries plus a merged `ensemble_metadata.json` (see step 5) that downstream training code can consume.

