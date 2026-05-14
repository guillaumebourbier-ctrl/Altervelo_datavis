#!/usr/bin/env python3
"""Synchronise observations SQL avec stations_enriched.csv (output de merge_csv.py).

INSERT OR IGNORE → les lignes historiques (déjà présentes) gardent leur
source='historical', seules les nouvelles arrivent avec source='live'.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DASHBOARD_DIR.parent
sys.path.insert(0, str(DASHBOARD_DIR))

from db import connect

ENRICHED_CSV = PROJECT_ROOT / "data" / "stations_enriched.csv"
TS_FMT = "%Y-%m-%dT%H:%M:%S%z"

OBS_COLS = [
    "timestamp", "station_index",
    "num_vehicles_available", "num_docks_available",
    "num_docks_disabled", "num_vehicles_disabled",
    "is_imputed", "n_vehicles_actifs", "n_vehicles_disabled_obs",
    "mean_battery", "min_battery", "max_battery", "pct_low_battery",
    "n_transit_0_150m", "n_transit_150_300m", "n_transit_300_450m",
    "n_transit_450_600m", "n_transit_600_750m", "n_transit_750_900m",
    "dist_nearest_transit_m", "is_obs_missing",
]

INSERT_SQL = f"""INSERT OR IGNORE INTO observations
    ({', '.join(OBS_COLS)}, source)
    VALUES ({', '.join(['?'] * len(OBS_COLS))}, 'live')"""


def main():
    t0 = time.time()
    df = pd.read_csv(ENRICHED_CSV, parse_dates=["timestamp"])
    df = df[OBS_COLS].copy()
    df["timestamp"] = df["timestamp"].dt.strftime(TS_FMT)
    rows = list(df.itertuples(index=False, name=None))

    conn = connect()
    cur = conn.cursor()
    cur.executemany(INSERT_SQL, rows)
    conn.commit()
    n_new = cur.rowcount
    n_total = cur.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
    n_live = cur.execute(
        "SELECT COUNT(*) FROM observations WHERE source='live'"
    ).fetchone()[0]
    conn.close()
    print(f"OK : +{n_new} obs (live total : {n_live:,} / {n_total:,}) "
          f"en {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
