#!/usr/bin/env bash
# Sync the COMPAS sparse-grid outputs between S3 and the local repository.

set -euo pipefail

S3_URI="${S3_URI:-s3://astrothesis-compas/ensemble_sparse}"
LOCAL_DIR="${LOCAL_DIR:-/Users/josephrodriguez/ASTROTHESIS/experiments/runs/compas_ensemble_sparse}"
MODE="${MODE:-down}"  # "down" (S3 -> local) or "up" (local -> S3)
AWS_EXTRA_SYNC_FLAGS="${AWS_EXTRA_SYNC_FLAGS:---only-show-errors}"

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

mkdir -p "${LOCAL_DIR}"

case "${MODE}" in
  down)
    log "Syncing from ${S3_URI} to ${LOCAL_DIR}"
    aws s3 sync "${S3_URI}/runs/" "${LOCAL_DIR}/" ${AWS_EXTRA_SYNC_FLAGS}
    aws s3 sync "${S3_URI}/metadata/" "${LOCAL_DIR}/metadata/" ${AWS_EXTRA_SYNC_FLAGS} || true
    ;;
  up)
    log "Syncing from ${LOCAL_DIR} to ${S3_URI}"
    aws s3 sync "${LOCAL_DIR}/" "${S3_URI}/runs/" ${AWS_EXTRA_SYNC_FLAGS}
    ;;
  *)
    echo "Unknown MODE=${MODE}. Use 'down' or 'up'." >&2
    exit 1
    ;;
esac

log "Sync complete"

