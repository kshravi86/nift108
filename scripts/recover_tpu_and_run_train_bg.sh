#!/bin/bash
set -euo pipefail

REPO_DIR="${1:-/home/Admin/nift108}"
DATA_URI="${2:-gs://nift108-bucket/nifty50_full_with_nakshatra_10am_mumbai.csv}"
GCS_PREFIX_BASE="${3:-gs://nift108-bucket/tpu-training-runs}"
SWAP_SIZE="${SWAP_SIZE:-8G}"

echo "[1/5] Stopping existing training process (if any)..."
pkill -f train_astrology_tpu.py || true

echo "[2/5] Restarting TPU runtime service..."
sudo systemctl restart tpu-runtime
sleep 3

echo "[3/5] Memory and swap before recovery:"
free -h || true
swapon --show || true

if ! swapon --noheadings --show=NAME | grep -q .; then
  echo "[4/5] No active swap found. Creating /swapfile (${SWAP_SIZE})..."
  if [ ! -f /swapfile ]; then
    sudo fallocate -l "${SWAP_SIZE}" /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
  fi
  sudo swapon /swapfile || true
else
  echo "[4/5] Swap already active. Skipping swap creation."
fi

echo "[5/5] Memory and swap after recovery:"
free -h || true
swapon --show || true

chmod +x "${REPO_DIR}/scripts/run_tpu_train_timestamp_bg.sh"
"${REPO_DIR}/scripts/run_tpu_train_timestamp_bg.sh" "${REPO_DIR}" "${DATA_URI}" "${GCS_PREFIX_BASE}"

echo "Follow logs with:"
echo "tail -f ${REPO_DIR}/tpu_train_*.log"
