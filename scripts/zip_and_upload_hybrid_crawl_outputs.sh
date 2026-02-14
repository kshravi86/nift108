#!/bin/bash
set -euo pipefail

# Zip hybrid crawl outputs (JSONL + state + logs) and upload the zip to GCS.
#
# Usage:
#   ./scripts/zip_and_upload_hybrid_crawl_outputs.sh [RUN_ID] [BUCKET] [PREFIX]
#
# Defaults:
# - RUN_ID: auto-detected from the newest gcp_docs_pmle_hybrid_*.jsonl in repo root
# - BUCKET: nift108-bucket
# - PREFIX: vertex-docs/zips/<RUN_ID>
#
# Example:
#   ./scripts/zip_and_upload_hybrid_crawl_outputs.sh hybrid_20260214_102048 nift108-bucket vertex-docs/zips/hybrid_20260214_102048

REPO_DIR="${REPO_DIR:-$HOME/nift108}"
RUN_ID="${1:-}"
BUCKET="${2:-nift108-bucket}"
PREFIX="${3:-}"

cd "${REPO_DIR}"

if [ -z "${RUN_ID}" ]; then
  latest="$(ls -t gcp_docs_pmle_hybrid_*.jsonl 2>/dev/null | head -n 1 || true)"
  if [ -z "${latest}" ]; then
    echo "ERROR: No files match gcp_docs_pmle_hybrid_*.jsonl in ${REPO_DIR}"
    exit 1
  fi
  base="$(basename "${latest}")"
  RUN_ID="${base#gcp_docs_pmle_hybrid_}"
  RUN_ID="${RUN_ID%.jsonl}"
fi

if [ -z "${PREFIX}" ]; then
  PREFIX="vertex-docs/zips/${RUN_ID}"
fi

if ! command -v zip >/dev/null 2>&1; then
  echo "zip not found; installing..."
  sudo apt-get update -y
  sudo apt-get install -y zip
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "${PYTHON_BIN}" ]; then
  if [ -x "${REPO_DIR}/.venv/bin/python" ]; then
    PYTHON_BIN="${REPO_DIR}/.venv/bin/python"
  else
    PYTHON_BIN="$(command -v python3 || command -v python)"
  fi
fi

# Ensure google-cloud-storage is available for the uploader.
if ! "${PYTHON_BIN}" -c "import google.cloud.storage" >/dev/null 2>&1; then
  "${PYTHON_BIN}" -m pip install -U google-cloud-storage
fi

OUT="gcp_docs_pmle_hybrid_${RUN_ID}.jsonl"
STATE="gcp_crawl_state_pmle_hybrid_${RUN_ID}.json"
ZIP="crawl_outputs_${RUN_ID}.zip"

files=()
for f in "${OUT}" "${STATE}"; do
  if [ -f "${f}" ]; then
    files+=("${f}")
  fi
done
for f in crawl_gcp_docs_hybrid_"${RUN_ID}".log*; do
  if [ -f "${f}" ]; then
    files+=("${f}")
  fi
done

if [ "${#files[@]}" -eq 0 ]; then
  echo "ERROR: No matching output/state/log files found for RUN_ID=${RUN_ID} in ${REPO_DIR}"
  exit 1
fi

rm -f "${ZIP}"
zip -r "${ZIP}" "${files[@]}"

echo "Uploading ${ZIP} -> gs://${BUCKET}/${PREFIX}/"
"${PYTHON_BIN}" upload_crawl_outputs_to_gcs.py \
  --gcs-prefix "gs://${BUCKET}/${PREFIX}/" \
  --workers 4 \
  "${ZIP}"

echo "Done."
