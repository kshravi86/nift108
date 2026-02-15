#!/usr/bin/env bash
set -euo pipefail

# All-in-one launcher for the hybrid GCP Docs crawler:
# - Ensures repo exists and is up to date
# - Runs VM setup (apt + venv + Playwright deps) in background
# - Waits until crawler starts, then prints the important paths
# - Optionally tails the crawler stdout log
#
# Usage:
#   ./scripts/run_hybrid_crawler_10k_all_in_one.sh
#   ./scripts/run_hybrid_crawler_10k_all_in_one.sh /home/Admin/nift108
#
# Env (optional):
#   HTTP_CONCURRENCY=20
#   PW_CONCURRENCY=2
#   MAX_PAGES=10000
#   FOLLOW=1              tail the crawler stdout log (Ctrl+C to stop tailing)
#   WAIT_FOR_START_SECS=900
#   SLEEP_SECS=2

REPO_URL="https://github.com/kshravi86/nift108.git"
REPO_DIR="${1:-$HOME/nift108}"

HTTP_CONCURRENCY="${HTTP_CONCURRENCY:-20}"
PW_CONCURRENCY="${PW_CONCURRENCY:-2}"
MAX_PAGES="${MAX_PAGES:-10000}"

FOLLOW="${FOLLOW:-1}"
WAIT_FOR_START_SECS="${WAIT_FOR_START_SECS:-900}"
SLEEP_SECS="${SLEEP_SECS:-2}"

need_cmd() {
  command -v "$1" >/dev/null 2>&1
}

ensure_git() {
  if need_cmd git; then
    return 0
  fi
  echo "Installing git..."
  sudo DEBIAN_FRONTEND=noninteractive NEEDRESTART_SUSPEND=1 NEEDRESTART_MODE=l \
    apt-get update -y
  sudo DEBIAN_FRONTEND=noninteractive NEEDRESTART_SUSPEND=1 NEEDRESTART_MODE=l \
    apt-get install -y git
}

ensure_repo() {
  if [ -d "${REPO_DIR}/.git" ]; then
    echo "Repo exists: ${REPO_DIR}"
    git -C "${REPO_DIR}" pull --ff-only
  else
    echo "Cloning repo into: ${REPO_DIR}"
    git clone "${REPO_URL}" "${REPO_DIR}"
  fi
}

parse_field() {
  local key="$1"
  local file="$2"
  grep -E "^${key}:" "${file}" 2>/dev/null | tail -n 1 | cut -d: -f2- | xargs || true
}

main() {
  ensure_git
  ensure_repo

  cd "${REPO_DIR}"
  chmod +x scripts/setup_vm_and_run_hybrid_crawler_bg.sh

  echo "Launching setup in background (will continue if you disconnect)..."
  LAUNCH_OUT="$(
    env HTTP_CONCURRENCY="${HTTP_CONCURRENCY}" PW_CONCURRENCY="${PW_CONCURRENCY}" MAX_PAGES="${MAX_PAGES}" \
      ./scripts/setup_vm_and_run_hybrid_crawler_bg.sh
  )"
  echo "${LAUNCH_OUT}"

  SETUP_LOG="$(echo "${LAUNCH_OUT}" | awk -F: '/^SETUP_LOG:/{print $2}' | tail -n 1 | xargs || true)"
  if [ -z "${SETUP_LOG}" ]; then
    echo "ERROR: Could not determine SETUP_LOG from launcher output."
    exit 1
  fi

  echo "Waiting for crawler to start (watching ${SETUP_LOG})..."
  start_ts="$(date +%s)"
  while true; do
    elapsed="$(( $(date +%s) - start_ts ))"
    if [ "${elapsed}" -ge "${WAIT_FOR_START_SECS}" ]; then
      echo "Timed out waiting for crawler to start after ${WAIT_FOR_START_SECS}s."
      echo "Check setup log:"
      echo "tail -n 200 \"${SETUP_LOG}\""
      exit 1
    fi

    STDOUT_LOG="$(parse_field "STDOUT_LOG" "${SETUP_LOG}")"
    OUTPUT_PATH="$(parse_field "OUTPUT" "${SETUP_LOG}")"
    PID="$(parse_field "STARTED_PID" "${SETUP_LOG}")"

    if [ -n "${STDOUT_LOG}" ] && [ -n "${OUTPUT_PATH}" ] && [ -f "${STDOUT_LOG}" ]; then
      echo "Crawler started."
      echo "PID:${PID}"
      echo "OUTPUT:${OUTPUT_PATH}"
      echo "STDOUT_LOG:${STDOUT_LOG}"
      break
    fi

    # Surface quick hints if setup failed early.
    if grep -qiE "FATAL:|Traceback|ERROR:" "${SETUP_LOG}" 2>/dev/null; then
      echo "Setup log shows an error; last 120 lines:"
      tail -n 120 "${SETUP_LOG}" || true
      exit 1
    fi

    echo "Not started yet... (${elapsed}s/${WAIT_FOR_START_SECS}s)"
    sleep "${SLEEP_SECS}"
  done

  echo "Sanity checks:"
  if [ -n "${PID}" ]; then
    ps -p "${PID}" -o pid,etime,cmd || true
  fi
  if [ -n "${OUTPUT_PATH}" ] && [ -f "${OUTPUT_PATH}" ]; then
    wc -l "${OUTPUT_PATH}" || true
  fi

  if [ "${FOLLOW}" = "1" ]; then
    echo "Tailing crawler log (Ctrl+C to stop tailing; crawler keeps running):"
    tail -f "${STDOUT_LOG}"
  else
    echo "FOLLOW=0: not tailing. You can follow logs with:"
    echo "tail -f \"${STDOUT_LOG}\""
  fi
}

main "$@"

