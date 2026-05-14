#!/usr/bin/env bash
# Orchestration cron 30min : collect → clean → merge → append SQL → predict → evaluate.
# À lancer depuis tmux : `while sleep 1800; do bash dashboard/pipeline.sh; done`
# ou via systemd timer (cf. README racine).

set -euo pipefail

cd "$(dirname "$0")/.."
PY=dashboard/.venv/bin/python

echo "=== $(date -Iseconds) ==="

echo "[1/6] collect.py"
$PY collect.py

echo "[2/6] cleaning_data.py"
$PY cleaning_data.py

echo "[3/6] clean_vehicle_status.py"
$PY clean_vehicle_status.py

echo "[4/6] merge_csv.py"
$PY merge_csv.py

echo "[5/6] append_live_obs.py"
$PY dashboard/append_live_obs.py

echo "[6/6] predict.py"
$PY dashboard/predict.py

echo "[6+/6] evaluate.py"
$PY dashboard/evaluate.py

echo "=== done ==="
