#!/usr/bin/env bash
# Run a subset of the COMPAS sparse-grid ensemble on EC2 and push results to S3.

set -euo pipefail

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

# --- Required inputs ---------------------------------------------------------
: "${START_INDEX:?Set START_INDEX to the inclusive grid index to run}"

# --- Tunable defaults --------------------------------------------------------
REPO_ROOT="${REPO_ROOT:-/opt/ASTROTHESIS}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/compas_runs}"
CONDA_PREFIX="${CONDA_PREFIX:-/opt/miniconda3}"
ENV_NAME="${ENV_NAME:-gw_channels}"
N_SYSTEMS="${N_SYSTEMS:-10000}"
CHUNK_SIZE="${CHUNK_SIZE:-1}"
END_INDEX="${END_INDEX:-$((START_INDEX + CHUNK_SIZE))}"
COMPAS_BINARY="${COMPAS_BINARY:-${REPO_ROOT}/simulators/compas/src/bin/COMPAS}"
S3_URI="${S3_URI:-s3://astrothesis-compas/ensemble_sparse}"
AWS_EXTRA_SYNC_FLAGS="${AWS_EXTRA_SYNC_FLAGS:---only-show-errors}"
OMP_THREADS="${OMP_THREADS:-4}"

if (( END_INDEX <= START_INDEX )); then
  echo "END_INDEX (${END_INDEX}) must be greater than START_INDEX (${START_INDEX})." >&2
  exit 1
fi

SLICE_LABEL=$(printf '%02d-%02d' "${START_INDEX}" "$((END_INDEX - 1))")
RUN_DIR="${OUTPUT_ROOT}/slice_${SLICE_LABEL}"
LOG_FILE="${RUN_DIR}/slice_${SLICE_LABEL}.log"
SUMMARY_FILE="${RUN_DIR}/slice_${SLICE_LABEL}.json"

conda_shell_init() {
  if [[ -f "${CONDA_PREFIX}/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "${CONDA_PREFIX}/etc/profile.d/conda.sh"
  else
    eval "$("${CONDA_PREFIX}/bin/conda" shell.bash hook)"
  fi
}

mkdir -p "${RUN_DIR}"
ulimit -n 65535 || true

log "Activating ${ENV_NAME}"
conda_shell_init
conda activate "${ENV_NAME}"

export OMP_NUM_THREADS="${OMP_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_THREADS}"
export MKL_NUM_THREADS="${OMP_THREADS}"
export RUN_DIR SUMMARY_FILE START_INDEX END_INDEX N_SYSTEMS SLICE_LABEL

log "Launching COMPAS sparse-grid slice ${SLICE_LABEL} (indices ${START_INDEX}:${END_INDEX})"
set -o pipefail
python -m pipelines.ensemble_generation.compas.generate_ensemble \
  --sparse \
  --n-systems "${N_SYSTEMS}" \
  --start-index "${START_INDEX}" \
  --end-index "${END_INDEX}" \
  --output-dir "${RUN_DIR}" \
  --compas-binary "${COMPAS_BINARY}" |& tee "${LOG_FILE}"

log "Generating slice summary"
python - <<'PY'
import json
import os
from datetime import datetime
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
meta_file = run_dir / "ensemble_metadata.json"
summary_file = Path(os.environ["SUMMARY_FILE"])

summary = {
    "start_index": int(os.environ["START_INDEX"]),
    "end_index": int(os.environ["END_INDEX"]),
    "n_systems": int(os.environ["N_SYSTEMS"]),
    "slice_label": os.environ["SLICE_LABEL"],
    "created_at": datetime.utcnow().isoformat() + "Z",
    "runs": [],
}

if meta_file.exists():
    data = json.loads(meta_file.read_text())
    summary["runs"] = data.get("runs", [])
    summary["n_runs"] = len(summary["runs"])
else:
    summary["n_runs"] = 0

summary_file.write_text(json.dumps(summary, indent=2))
PY

log "Syncing slice outputs to ${S3_URI}/runs/slice_${SLICE_LABEL}/"
aws s3 sync "${RUN_DIR}/" "${S3_URI}/runs/slice_${SLICE_LABEL}/" ${AWS_EXTRA_SYNC_FLAGS}

log "Uploading slice summary and logs"
aws s3 cp "${SUMMARY_FILE}" "${S3_URI}/metadata/slice_${SLICE_LABEL}.json" ${AWS_EXTRA_SYNC_FLAGS}
aws s3 cp "${LOG_FILE}" "${S3_URI}/logs/slice_${SLICE_LABEL}.log" ${AWS_EXTRA_SYNC_FLAGS}

log "Slice ${SLICE_LABEL} complete"

