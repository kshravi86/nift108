#!/bin/bash
set -euo pipefail

REPO_DIR="${1:-$HOME/nift108}"
LOG_PATH="${2:-$HOME/nift108/tpu_train.log}"
OUT_DIR="${3:-$HOME/nift108/tpu_outputs/latest}"
DATA_URI="${4:-gs://nift108-bucket/nifty50_full_with_nakshatra_10am_mumbai.csv}"
GCS_OUT_PREFIX="${5:-gs://nift108-bucket/tpu-training-runs/latest}"

cd "$REPO_DIR"

python3 - <<'PY'
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("pandas") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "pandas"])
PY

nohup python3 "$REPO_DIR/tpu/train_astrology_tpu.py" \
  --data-uri "$DATA_URI" \
  --output-dir "$OUT_DIR" \
  --gcs-output-prefix "$GCS_OUT_PREFIX" \
  > "$LOG_PATH" 2>&1 < /dev/null &

echo "STARTED_PID:$!"
echo "LOG:$LOG_PATH"
