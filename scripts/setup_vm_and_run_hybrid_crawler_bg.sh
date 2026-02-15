#!/bin/bash
set -euo pipefail

# One-shot Ubuntu VM setup + hybrid crawler launch (background) + log tail.
#
# Usage (defaults):
#   ./scripts/setup_vm_and_run_hybrid_crawler_bg.sh
#
# Optional env vars:
#   DETACH=1 (default) to background this setup script itself and write logs to SETUP_LOG
#   SETUP_LOG=/path/to/setup.log (only used when DETACH=1)
#   TAIL=1 (default when DETACH=0) to tail crawler stdout log; set TAIL=0 to skip tailing
#   WAIT_SECS=900 how long to wait for apt/dpkg locks before failing (0 disables waiting)
#   SLEEP_SECS=5  sleep interval while waiting for apt/dpkg locks
#   REPO_DIR=/home/<user>/nift108
#   HTTP_CONCURRENCY=20
#   PW_CONCURRENCY=2
#   MAX_PAGES=10000

REPO_URL="https://github.com/kshravi86/nift108.git"
REPO_DIR="${REPO_DIR:-$HOME/nift108}"
HTTP_CONCURRENCY="${HTTP_CONCURRENCY:-20}"
PW_CONCURRENCY="${PW_CONCURRENCY:-2}"
MAX_PAGES="${MAX_PAGES:-10000}"

# By default, detach this setup script so you can close SSH and let the job run.
# To run in the foreground instead: DETACH=0 ./scripts/setup_vm_and_run_hybrid_crawler_bg.sh
DETACH="${DETACH:-1}"
if [[ "${DETACH}" == "1" && -z "${_SETUP_DETACHED:-}" && -t 1 ]]; then
  SETUP_RUN_ID="${SETUP_RUN_ID:-setup_$(date +%Y%m%d_%H%M%S)}"
  SETUP_LOG="${SETUP_LOG:-$HOME/${SETUP_RUN_ID}.log}"
  # Re-run in background. Disable tailing in detached mode.
  nohup env _SETUP_DETACHED=1 DETACH=0 TAIL=0 \
    "$0" "$@" > "${SETUP_LOG}" 2>&1 < /dev/null &
  PID="$!"
  echo "STARTED_SETUP_PID:${PID}"
  echo "SETUP_LOG:${SETUP_LOG}"
  echo "Tail:"
  echo "tail -f \"${SETUP_LOG}\""
  exit 0
fi

# Avoid interactive apt prompts (e.g., needrestart "restart services?" dialog).
# Ref: needrestart(1) supports NEEDRESTART_SUSPEND/NEEDRESTART_MODE.
DEBIAN_FRONTEND="${DEBIAN_FRONTEND:-noninteractive}"
NEEDRESTART_SUSPEND="${NEEDRESTART_SUSPEND:-1}"
NEEDRESTART_MODE="${NEEDRESTART_MODE:-l}"

WAIT_SECS="${WAIT_SECS:-900}"
SLEEP_SECS="${SLEEP_SECS:-5}"

LOCKS=(
  "/var/lib/dpkg/lock-frontend"
  "/var/lib/dpkg/lock"
  "/var/lib/apt/lists/lock"
)

has_lock() {
  for l in "${LOCKS[@]}"; do
    if sudo fuser "$l" >/dev/null 2>&1; then
      return 0
    fi
  done
  return 1
}

wait_for_apt() {
  if [ "${WAIT_SECS}" = "0" ]; then
    return 0
  fi
  local start_ts
  start_ts="$(date +%s)"
  while has_lock || pgrep -x apt-get >/dev/null || pgrep -x dpkg >/dev/null || pgrep -x unattended-upgrade >/dev/null; do
    local elapsed
    elapsed="$(( $(date +%s) - start_ts ))"
    echo "Waiting for apt/dpkg locks... (${elapsed}s/${WAIT_SECS}s)"
    if [ "${elapsed}" -ge "${WAIT_SECS}" ]; then
      echo "Timed out waiting for apt/dpkg locks. Re-run later or set WAIT_SECS higher."
      exit 1
    fi
    sleep "${SLEEP_SECS}"
  done
}

echo "[1/6] Installing system packages..."
wait_for_apt
sudo DEBIAN_FRONTEND="${DEBIAN_FRONTEND}" NEEDRESTART_SUSPEND="${NEEDRESTART_SUSPEND}" NEEDRESTART_MODE="${NEEDRESTART_MODE}" \
  apt-get update -y
wait_for_apt
sudo DEBIAN_FRONTEND="${DEBIAN_FRONTEND}" NEEDRESTART_SUSPEND="${NEEDRESTART_SUSPEND}" NEEDRESTART_MODE="${NEEDRESTART_MODE}" \
  apt-get install -y git python3-pip python3-venv

echo "[2/6] Cloning/updating repo..."
if [ -d "${REPO_DIR}/.git" ]; then
  git -C "${REPO_DIR}" pull --ff-only
else
  git clone "${REPO_URL}" "${REPO_DIR}"
fi

echo "[3/6] Creating venv + installing Python deps..."
cd "${REPO_DIR}"
python3 -m venv .venv
VENV_PY="${REPO_DIR}/.venv/bin/python"
"${VENV_PY}" -m pip install --upgrade pip
"${VENV_PY}" -m pip install httpx beautifulsoup4 playwright

echo "[4/6] Installing Playwright system deps + Chromium..."
wait_for_apt
sudo DEBIAN_FRONTEND="${DEBIAN_FRONTEND}" NEEDRESTART_SUSPEND="${NEEDRESTART_SUSPEND}" NEEDRESTART_MODE="${NEEDRESTART_MODE}" \
  "${VENV_PY}" -m playwright install-deps chromium
"${VENV_PY}" -m playwright install chromium

echo "[5/6] Launching crawler in background..."
chmod +x scripts/run_hybrid_crawler_bg.sh
LAUNCH_OUT="$(PYTHON_BIN="${VENV_PY}" ./scripts/run_hybrid_crawler_bg.sh "${REPO_DIR}" "${HTTP_CONCURRENCY}" "${PW_CONCURRENCY}" "${MAX_PAGES}")"
echo "${LAUNCH_OUT}"

TAIL="${TAIL:-1}"
if [[ "${TAIL}" == "0" || "${TAIL}" == "false" || "${TAIL}" == "no" ]]; then
  echo "[6/6] Skipping log tail (TAIL=${TAIL})."
  STDOUT_LOG="$(echo "${LAUNCH_OUT}" | awk -F: '/^STDOUT_LOG:/{print $2}' | tail -n 1 | xargs || true)"
  if [ -n "${STDOUT_LOG}" ]; then
    echo "Follow logs with:"
    echo "tail -f \"${STDOUT_LOG}\""
  fi
  exit 0
fi

echo "[6/6] Tailing latest log..."
STDOUT_LOG="$(echo "${LAUNCH_OUT}" | awk -F: '/^STDOUT_LOG:/{print $2}' | tail -n 1 | xargs || true)"
if [ -z "${STDOUT_LOG}" ] || [ ! -f "${STDOUT_LOG}" ]; then
  STDOUT_LOG="$(ls -t "${REPO_DIR}"/crawl_gcp_docs_hybrid_*.log.stdout 2>/dev/null | head -n 1 || true)"
fi
if [ -z "${STDOUT_LOG}" ]; then
  echo "No stdout log found. Check ${REPO_DIR} for crawl_gcp_docs_hybrid_*.log.stdout"
  exit 1
fi
echo "Tailing: ${STDOUT_LOG}"
tail -f "${STDOUT_LOG}"
