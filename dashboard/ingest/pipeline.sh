#!/usr/bin/env bash
# Pipeline d'ingestion DB-first AlterVélo.
# Entièrement contenu dans dashboard/ — ne lit/écrit jamais hors de ce dossier.
#
# Tick = 1 cycle (collect → clean → fill_gaps → clean_vehicles → merge → predict → evaluate).
# Pour mettre en boucle 30 min, voir README.md (section « Production »).

set -euo pipefail

cd "$(dirname "$0")/.."   # → dashboard/
PY=.venv/bin/python

echo "=== $(date -Iseconds) ==="
$PY ingest/collect.py            # APIs       → raw_*_status
$PY ingest/clean_stations.py     # raw_status → stations_clean (is_imputed=0)
$PY ingest/fill_gaps.py          # stations_clean (insert imputés)
$PY ingest/clean_vehicles.py     # raw_vehicle_status → vehicles_clean
$PY ingest/merge.py              # stations_clean + vehicles_clean → observations
$PY predict.py                   # observations → predictions (4 horizons)
$PY evaluate.py                  # backfill y_obs / err_*
echo "=== done ==="
