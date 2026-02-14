#!/bin/bash
set -euo pipefail

# Runs the hybrid GCP docs crawler (httpx + Playwright fallback) in the background with logs.
#
# Usage:
#   ./scripts/run_hybrid_crawler_bg.sh [REPO_DIR] [HTTP_CONCURRENCY] [PW_CONCURRENCY] [MAX_PAGES]
#
# Example:
#   ./scripts/run_hybrid_crawler_bg.sh /home/Admin/nift108 15 2 2000

REPO_DIR="${1:-$HOME/nift108}"
HTTP_CONCURRENCY="${2:-10}"
PW_CONCURRENCY="${3:-2}"
MAX_PAGES="${4:-2000}"

RUN_ID="${RUN_ID:-hybrid_$(date +%Y%m%d_%H%M%S)}"

OUTPUT_PATH="${OUTPUT_PATH:-$REPO_DIR/gcp_docs_pmle_hybrid_${RUN_ID}.jsonl}"
STATE_PATH="${STATE_PATH:-$REPO_DIR/gcp_crawl_state_pmle_hybrid_${RUN_ID}.json}"
LOG_PATH="${LOG_PATH:-$REPO_DIR/crawl_gcp_docs_hybrid_${RUN_ID}.log}"

CRAWLER="${CRAWLER:-$REPO_DIR/crawl_gcp_docs_hybrid_httpx_playwright.py}"

cd "$REPO_DIR"

nohup env PYTHONUNBUFFERED=1 python3 "$CRAWLER" \
  --http-concurrency "$HTTP_CONCURRENCY" \
  --playwright-concurrency "$PW_CONCURRENCY" \
  --max-pages "$MAX_PAGES" \
  --output-path "$OUTPUT_PATH" \
  --state-path "$STATE_PATH" \
  --log-path "$LOG_PATH" \
  > "${LOG_PATH}.stdout" 2>&1 < /dev/null &

PID="$!"
echo "STARTED_PID:$PID"
echo "RUN_ID:$RUN_ID"
echo "OUTPUT:$OUTPUT_PATH"
echo "STATE:$STATE_PATH"
echo "LOG:$LOG_PATH"
echo "STDOUT_LOG:${LOG_PATH}.stdout"
