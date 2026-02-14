#!/bin/bash
set -euo pipefail

REPO_DIR="${1:-/home/Admin/nift108}"
DATA_URI="${2:-gs://nift108-bucket/nifty50_full_with_nakshatra_10am_mumbai.csv}"
GCS_PREFIX_BASE="${3:-gs://nift108-bucket/tpu-training-runs}"

echo "[1/6] Stopping existing training process (if any)..."
pkill -f train_astrology_tpu.py || true

echo "[2/6] Reinstalling TensorFlow dependency pins (user site)..."
python3 -m pip install --user --upgrade --force-reinstall \
  "protobuf==4.25.3" \
  "numpy==1.26.4"

echo "[3/6] Verifying TensorFlow/protobuf/numpy imports..."
python3 -c "import tensorflow as tf, google.protobuf as p, numpy as np; print('tf', tf.__version__, 'protobuf', p.__version__, 'numpy', np.__version__)"

echo "[4/6] Relaxing shell memory ulimits (best effort)..."
ulimit -v unlimited || true
ulimit -m unlimited || true

echo "[5/6] Running TPU recovery + background launcher..."
chmod +x "${REPO_DIR}/scripts/recover_tpu_and_run_train_bg.sh"
"${REPO_DIR}/scripts/recover_tpu_and_run_train_bg.sh" "${REPO_DIR}" "${DATA_URI}" "${GCS_PREFIX_BASE}"

echo "[6/6] Done."
