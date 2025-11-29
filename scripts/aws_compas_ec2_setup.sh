#!/usr/bin/env bash
# Bootstrap an EC2 instance so it can run the COMPAS sparse-grid ensemble.
# This script is idempotent and can be used as user data or run interactively.

set -euo pipefail

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

# --- Tunable defaults --------------------------------------------------------
REPO_URL="${REPO_URL:-https://github.com/josephrodriguez/ASTROTHESIS.git}"
GIT_REF="${GIT_REF:-main}"
REPO_ROOT="${REPO_ROOT:-/opt/ASTROTHESIS}"
CONDA_PREFIX="${CONDA_PREFIX:-/opt/miniconda3}"
ENV_NAME="${ENV_NAME:-gw_channels}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
APT_PACKAGES="${APT_PACKAGES:-build-essential cmake git curl unzip libgsl-dev libhdf5-dev libboost-all-dev libyaml-cpp-dev awscli}"
RUN_ROOT="${RUN_ROOT:-/mnt/compas_runs}"

# --- Helper functions --------------------------------------------------------
ensure_conda() {
  if [[ -x "${CONDA_PREFIX}/bin/conda" ]]; then
    return
  fi

  log "Installing Miniconda at ${CONDA_PREFIX}"
  tmp_installer="$(mktemp)"
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "${tmp_installer}"
  bash "${tmp_installer}" -b -p "${CONDA_PREFIX}"
  rm -f "${tmp_installer}"
}

conda_shell_init() {
  if [[ -f "${CONDA_PREFIX}/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "${CONDA_PREFIX}/etc/profile.d/conda.sh"
  else
    eval "$("${CONDA_PREFIX}/bin/conda" shell.bash hook)"
  fi
}

ensure_repo() {
  if [[ -d "${REPO_ROOT}/.git" ]]; then
    log "Repository already exists, fetching latest ${GIT_REF}"
    git -C "${REPO_ROOT}" fetch --all --prune
    git -C "${REPO_ROOT}" checkout "${GIT_REF}"
    git -C "${REPO_ROOT}" pull --ff-only origin "${GIT_REF}"
    return
  fi

  log "Cloning repository ${REPO_URL} into ${REPO_ROOT}"
  sudo mkdir -p "$(dirname "${REPO_ROOT}")"
  sudo chown "$(whoami)":"$(whoami)" "$(dirname "${REPO_ROOT}")"
  git clone "${REPO_URL}" "${REPO_ROOT}"
  git -C "${REPO_ROOT}" checkout "${GIT_REF}"
}

ensure_run_root() {
  if [[ -d "${RUN_ROOT}" ]]; then
    return
  fi

  log "Creating ${RUN_ROOT} for COMPAS outputs"
  sudo mkdir -p "${RUN_ROOT}"
  sudo chown "$(whoami)":"$(whoami)" "${RUN_ROOT}"
}

build_compas() {
  log "Compiling COMPAS (this can take several minutes)"
  pushd "${REPO_ROOT}/simulators/compas/src" >/dev/null
  make -j"$(nproc)" fast
  popd >/dev/null
}

# --- Execution ---------------------------------------------------------------
log "Updating apt cache and installing base packages"
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y ${APT_PACKAGES}

ensure_repo
ensure_conda
conda_shell_init

if ! conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -qx "${ENV_NAME}"; then
  log "Creating ${ENV_NAME} conda environment (Python ${PYTHON_VERSION})"
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

log "Activating ${ENV_NAME}"
conda activate "${ENV_NAME}"
python -m pip install --upgrade pip
python -m pip install -r "${REPO_ROOT}/configs/infrastructure/requirements.txt"

build_compas
ensure_run_root

log "Ensuring AWS credentials are configured (aws configure)"
if ! aws sts get-caller-identity >/dev/null 2>&1; then
  log "AWS CLI not configured. Run 'aws configure' with an IAM user/role that can access s3://astrothesis-compas/"
fi

log "Bootstrap complete. Use scripts/aws_compas_run_slice.sh to launch ensemble slices."

