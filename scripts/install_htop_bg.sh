#!/bin/bash
set -euo pipefail
while pgrep -x apt-get >/dev/null || pgrep -x dpkg >/dev/null; do
  sleep 5
done
sudo apt-get update
sudo apt-get install -y htop
htop --version > /home/Admin/htop_version.txt
