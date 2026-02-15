#!/bin/bash
set -euo pipefail

# One-shot Ubuntu VM setup + hybrid crawler launch (background) + log tail.
#
# Usage (defaults):
#   ./scripts/setup_vm_and_run_hybrid_crawler_bg.sh
#
# Optional env vars:
#   REPO_DIR=/home/<user>/nift108
#   HTTP_CONCURRENCY=20
#   PW_CONCURRENCY=2
#   MAX_PAGES=10000

REPO_URL="https://github.com/kshravi86/nift108.git"
REPO_DIR="${REPO_DIR:-$HOME/nift108}"
HTTP_CONCURRENCY="${HTTP_CONCURRENCY:-20}"
PW_CONCURRENCY="${PW_CONCURRENCY:-2}"
MAX_PAGES="${MAX_PAGES:-10000}"

# Avoid interactive apt prompts (e.g., needrestart "restart services?" dialog).
# Ref: needrestart(1) supports NEEDRESTART_SUSPEND/NEEDRESTART_MODE.
DEBIAN_FRONTEND="${DEBIAN_FRONTEND:-noninteractive}"
NEEDRESTART_SUSPEND="${NEEDRESTART_SUSPEND:-1}"
NEEDRESTART_MODE="${NEEDRESTART_MODE:-l}"

echo "[1/6] Installing system packages..."
sudo DEBIAN_FRONTEND="${DEBIAN_FRONTEND}" NEEDRESTART_SUSPEND="${NEEDRESTART_SUSPEND}" NEEDRESTART_MODE="${NEEDRESTART_MODE}" \
  apt-get update -y
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
sudo DEBIAN_FRONTEND="${DEBIAN_FRONTEND}" NEEDRESTART_SUSPEND="${NEEDRESTART_SUSPEND}" NEEDRESTART_MODE="${NEEDRESTART_MODE}" \
  "${VENV_PY}" -m playwright install-deps chromium
"${VENV_PY}" -m playwright install chromium

echo "[5/6] Launching crawler in background..."
chmod +x scripts/run_hybrid_crawler_bg.sh
LAUNCH_OUT="$(PYTHON_BIN="${VENV_PY}" ./scripts/run_hybrid_crawler_bg.sh "${REPO_DIR}" "${HTTP_CONCURRENCY}" "${PW_CONCURRENCY}" "${MAX_PAGES}")"
echo "${LAUNCH_OUT}"

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
