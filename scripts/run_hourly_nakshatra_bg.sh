#!/usr/bin/env bash
set -euo pipefail

# Runs hourly nakshatra generation as a background job.
#
# Usage:
#   ./scripts/run_hourly_nakshatra_bg.sh [REPO_ROOT] [INPUT_CSV] [OUTPUT_CSV]
#
# Env:
#   PYTHON_BIN: path to python to use (optional; otherwise uses .venv/bin/python if present)

ROOT="${1:-$HOME/nift108}"
INPUT_CSV="${2:-$ROOT/nifty50_hourly_last_3_years.csv}"
OUTPUT_CSV="${3:-$ROOT/nifty50_hourly_with_nakshatra_mumbai_lahiri.csv}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY="$PYTHON_BIN"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="python3"
fi

cd "$ROOT"
mkdir -p "$ROOT/logs"

RUN_ID="hourly_nak_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$ROOT/logs/nifty50_hourly_nakshatra_${RUN_ID}.log"
STDOUT_LOG="${LOG_FILE}.stdout"

echo "Using python: $PY"
echo "Input:  $INPUT_CSV"
echo "Output: $OUTPUT_CSV"

echo "Installing deps (best effort)..."
"$PY" -m pip install --upgrade pip >/dev/null 2>&1 || true
"$PY" -m pip install -r requirements.txt >/dev/null

nohup "$PY" -u generate_nakshatra_for_nifty_hourly.py \
  --nifty-csv "$INPUT_CSV" \
  --output "$OUTPUT_CSV" \
  --log-file "$LOG_FILE" \
  > "$STDOUT_LOG" 2>&1 &

PID=$!
echo "STARTED_PID:$PID"
echo "RUN_ID:$RUN_ID"
echo "LOG:$LOG_FILE"
echo "STDOUT_LOG:$STDOUT_LOG"
echo "Tail:"
echo "tail -f \"$STDOUT_LOG\""

