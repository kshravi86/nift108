#!/bin/bash
set -euo pipefail

REPO_DIR="${1:-/home/Admin/nift108}"
DATA_URI="${2:-gs://nift108-bucket/nifty50_full_with_nakshatra_10am_mumbai.csv}"
GCS_PREFIX_BASE="${3:-gs://nift108-bucket/tpu-training-runs}"

RUN_ID="${RUN_ID:-run_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="$REPO_DIR/tpu_outputs/$RUN_ID"
LOG_FILE="$REPO_DIR/tpu_train_${RUN_ID}.log"
GCS_OUT="$GCS_PREFIX_BASE/$RUN_ID"

mkdir -p "$REPO_DIR/tpu_outputs"

nohup python3 "$REPO_DIR/tpu/train_astrology_tpu.py" \
  --data-uri "$DATA_URI" \
  --output-dir "$OUT_DIR" \
  --gcs-output-prefix "$GCS_OUT" \
  > "$LOG_FILE" 2>&1 < /dev/null &

PID="$!"
echo "STARTED_PID:$PID"
echo "RUN_ID:$RUN_ID"
echo "LOG:$LOG_FILE"
echo "OUT_DIR:$OUT_DIR"
echo "GCS_OUT:$GCS_OUT"
