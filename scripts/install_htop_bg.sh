#!/bin/bash
set -euo pipefail

# Installs htop while safely waiting for any other apt/dpkg process to finish.
#
# Env:
#   WAIT_SECS=600        How long to wait for apt/dpkg locks before giving up
#   SLEEP_SECS=5         Sleep interval while waiting
#   FORCE_KILL=0         If 1, tries to kill lock-holding PIDs after WAIT_SECS and repair dpkg
#   RUN_UPDATE=1         If 0, skip apt-get update

WAIT_SECS="${WAIT_SECS:-600}"
SLEEP_SECS="${SLEEP_SECS:-5}"
FORCE_KILL="${FORCE_KILL:-0}"
RUN_UPDATE="${RUN_UPDATE:-1}"

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

lock_pids() {
  local pids=""
  for l in "${LOCKS[@]}"; do
    pids+=" $(sudo fuser "$l" 2>/dev/null || true)"
  done
  echo "$pids" | tr ' ' '\n' | awk 'NF{print $1}' | sort -n | uniq
}

start_ts="$(date +%s)"
while has_lock || pgrep -x apt-get >/dev/null || pgrep -x dpkg >/dev/null || pgrep -x unattended-upgrade >/dev/null; do
  elapsed="$(( $(date +%s) - start_ts ))"
  echo "Waiting for apt/dpkg to finish... (${elapsed}s/${WAIT_SECS}s)"
  if [ "$elapsed" -ge "$WAIT_SECS" ]; then
    echo "Timed out waiting for dpkg/apt locks."
    if [ "$FORCE_KILL" != "1" ]; then
      echo "Re-run with FORCE_KILL=1 to kill lock holder(s) and attempt dpkg repair."
      exit 1
    fi
    echo "FORCE_KILL=1: killing lock holder(s) (best effort)..."
    for pid in $(lock_pids); do
      if [ -n "$pid" ]; then
        sudo kill "$pid" 2>/dev/null || true
      fi
    done
    sleep 3
    for pid in $(lock_pids); do
      if [ -n "$pid" ]; then
        sudo kill -9 "$pid" 2>/dev/null || true
      fi
    done
    echo "Repairing dpkg (best effort)..."
    sudo dpkg --configure -a || true
    sudo apt-get -f install -y || true
    break
  fi
  sleep "$SLEEP_SECS"
done

DEBIAN_FRONTEND="${DEBIAN_FRONTEND:-noninteractive}"
NEEDRESTART_SUSPEND="${NEEDRESTART_SUSPEND:-1}"
NEEDRESTART_MODE="${NEEDRESTART_MODE:-l}"

if [ "$RUN_UPDATE" = "1" ]; then
  sudo DEBIAN_FRONTEND="${DEBIAN_FRONTEND}" NEEDRESTART_SUSPEND="${NEEDRESTART_SUSPEND}" NEEDRESTART_MODE="${NEEDRESTART_MODE}" \
    apt-get update -y
fi

sudo DEBIAN_FRONTEND="${DEBIAN_FRONTEND}" NEEDRESTART_SUSPEND="${NEEDRESTART_SUSPEND}" NEEDRESTART_MODE="${NEEDRESTART_MODE}" \
  apt-get install -y htop

htop --version | tee "$HOME/htop_version.txt" >/dev/null
echo "OK: htop installed"
