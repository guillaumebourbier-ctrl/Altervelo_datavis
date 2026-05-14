# `dashboard/ingest/` — pipeline d'ingestion DB-first

Pipeline d'alimentation 100 % SQLite, autonome dans `dashboard/`. Remplace la
chaîne CSV historique (`../collect.py` → `../cleaning_data.py` →
`../clean_vehicle_status.py` → `../merge_csv.py` → `append_live_obs.py`).

Aucun script ici ne lit/écrit en dehors de `dashboard/`. Tout transite par
`velos.db`.

## Étapes (chacune idempotente)

| # | Script | Lit | Écrit |
|---|--------|-----|-------|
| 1 | `collect.py` | API GBFS | `raw_station_status`, `raw_vehicle_status` (INSERT OR IGNORE) |
| 2 | `clean_stations.py` | `raw_station_status` | `stations_clean` (is_imputed=0, INSERT OR REPLACE) |
| 3 | `fill_gaps.py` | `stations_clean` | `stations_clean` (is_imputed=1, INSERT OR IGNORE) |
| 4 | `clean_vehicles.py` | `raw_vehicle_status`, `stations_clean` | `vehicles_clean` (INSERT OR REPLACE) |
| 5 | `merge.py` | `stations_clean`, `vehicles_clean` | `observations` (source='live', INSERT OR IGNORE) |
| 6 | `../predict.py` | `observations` | `predictions` (source='live') |
| 7 | `../evaluate.py` | `observations`, `predictions` | `predictions` (UPDATE y_obs/err_*) |

Chaque étape logge durée + statut dans la table `pipeline_runs`.

## Bootstrap

```bash
cd /home/dok/3A/Lab/Data/Fouille/Velos/dashboard
.venv/bin/pip install -r requirements.txt        # ajoute requests
.venv/bin/python ingest/schema.py                # crée raw_* + *_clean (idempotent)
```

## Tick unique

```bash
bash dashboard/ingest/pipeline.sh
```

Sortie attendue :
```
=== 2026-05-04T14:05:23+04:00 ===
OK collect — ts=...  stations:+31  vehicles:+128
OK clean_stations — 31 lignes (INSERT OR REPLACE, is_imputed=0)
OK fill_gaps — 0 lignes imputées insérées        # ou >0 si trous
OK clean_vehicles — 124 lignes (INSERT OR REPLACE)
OK merge — +31 obs (source='live')
... (predict + evaluate)
=== done ===
```

## Production (boucle background, 30 min)

```bash
nohup bash -c 'while true; do bash dashboard/ingest/pipeline.sh; sleep 1800; done' \
    >> dashboard/ingest/pipeline.log 2>&1 &
disown
```

Inspecter :
```bash
tail -f dashboard/ingest/pipeline.log
sqlite3 dashboard/velos.db \
  "SELECT ts_run, status, duration_ms, n_obs_inserted FROM pipeline_runs ORDER BY ts_run DESC LIMIT 12;"
```

Arrêter :
```bash
pkill -f 'dashboard/ingest/pipeline.sh'
```

### Alternative systemd (timer utilisateur)

`~/.config/systemd/user/altervelo-ingest.service`
```ini
[Unit]
Description=AlterVélo ingest tick

[Service]
Type=oneshot
WorkingDirectory=%h/3A/Lab/Data/Fouille/Velos
ExecStart=/usr/bin/bash %h/3A/Lab/Data/Fouille/Velos/dashboard/ingest/pipeline.sh
```

`~/.config/systemd/user/altervelo-ingest.timer`
```ini
[Unit]
Description=Run AlterVélo ingest every 30 min

[Timer]
OnBootSec=2min
OnUnitActiveSec=30min
Persistent=true

[Install]
WantedBy=timers.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now altervelo-ingest.timer
systemctl --user list-timers
```

## Schéma des nouvelles tables

```sql
raw_station_status(ts_collect, station_id, is_installed, is_renting, is_returning,
                   num_docks_available, num_docks_disabled,
                   num_vehicles_available, num_vehicles_disabled,
                   last_reported, vehicle_types_available)
                   PK (ts_collect, station_id)

raw_vehicle_status(ts_collect, vehicle_id, vehicle_type_id, lat, lon,
                   current_fuel_percent, current_range_meters,
                   is_disabled, is_reserved, last_reported, station_id)
                   PK (ts_collect, vehicle_id)

stations_clean(timestamp, station_index, station_name, lat, lon, capacity,
               num_docks_available, num_docks_disabled,
               num_vehicles_available, num_vehicles_disabled,
               count_x2, is_imputed)
               PK (timestamp, station_index)

vehicles_clean(timestamp, vehicle_id, station_index,
               lat, lon, current_fuel_percent, current_range_meters, is_disabled)
               PK (timestamp, vehicle_id)
```

Toutes append-only ; rétention illimitée (cf. décision projet : audit complet).
Pour purger les imputations en vue d'une re-imputation après arrivée de raw :
```sql
DELETE FROM stations_clean WHERE is_imputed = 1;
```
puis relancer `fill_gaps.py`.

## Invariant train/serve

Le tuple `OBS_COLS` dans `merge.py` doit rester *byte-identique* à
`../append_live_obs.py:OBS_COLS` et à l'ordre attendu par
`../train_xgboostplus_delta.py`. Toute divergence = train/serve skew silencieux.
Cf. `dashboard/CLAUDE.md` section « Critical invariant ».
