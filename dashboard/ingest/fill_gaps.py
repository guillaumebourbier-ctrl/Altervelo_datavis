#!/usr/bin/env python3
"""Étape 3 — Comble les slots manquants dans stations_clean.

Construit la grille canonique (slot 30 min × station_index) sur la fenêtre
[min, max] de stations_clean, et insère les slots absents avec :
  - nuit (20h-5h locale)  : forward-fill
  - jour (5h-20h)         : interpolation linéaire des compteurs num_*, ffill du reste

Marque les lignes synthétiques avec is_imputed=1.

Idempotent : INSERT OR IGNORE — n'écrase JAMAIS une vraie observation
(is_imputed=0) ni une imputation déjà présente. Si on veut "re-imputer" après
arrivée de nouvelles raw, supprimer d'abord les lignes is_imputed=1 :
  DELETE FROM stations_clean WHERE is_imputed=1;
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from db import connect  # noqa: E402

from _common import GRID_MIN, TZ_REUNION, pipeline_step  # noqa: E402

NUMERIC_COLS = [
    "num_docks_available", "num_docks_disabled",
    "num_vehicles_available", "num_vehicles_disabled",
]
NIGHT_HOURS = set(list(range(20, 24)) + list(range(0, 5)))
TS_FMT = "%Y-%m-%dT%H:%M:%S%z"

INSERT_SQL = """INSERT OR IGNORE INTO stations_clean
    (timestamp, station_index, station_name, lat, lon, capacity,
     num_docks_available, num_docks_disabled,
     num_vehicles_available, num_vehicles_disabled,
     count_x2, is_imputed)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)"""


def fill(conn) -> int:
    df = pd.read_sql("SELECT * FROM stations_clean", conn,
                     parse_dates=["timestamp"])
    if df.empty:
        return 0

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(TZ_REUNION)

    # Grille canonique
    slots = pd.date_range(df["timestamp"].min(), df["timestamp"].max(),
                          freq=f"{GRID_MIN}min", tz=TZ_REUNION)
    stations = (df[["station_index", "station_name", "lat", "lon", "capacity"]]
                .drop_duplicates("station_index"))
    grid = stations.merge(pd.DataFrame({"timestamp": slots}), how="cross")

    merged = grid.merge(
        df[["timestamp", "station_index", *NUMERIC_COLS, "count_x2", "is_imputed"]],
        on=["timestamp", "station_index"], how="left",
    ).sort_values(["station_index", "timestamp"]).reset_index(drop=True)

    is_new = merged["is_imputed"].isna()  # slots à imputer
    if not is_new.any():
        return 0

    local_hour = merged["timestamp"].dt.hour
    is_night = local_hour.isin(NIGHT_HOURS)

    grouped = merged.groupby("station_index", group_keys=False)
    ffilled = grouped.ffill()
    # Interpolation linéaire de jour (compteurs uniquement)
    interp = grouped[NUMERIC_COLS].apply(
        lambda g: g.interpolate(method="linear", limit_direction="both")
    )
    for col in NUMERIC_COLS:
        merged[col] = np.where(is_night, ffilled[col], interp[col])
    # count_x2 : ffill puis bfill
    merged["count_x2"] = grouped["count_x2"].ffill().bfill()

    new_rows = merged[is_new].copy()
    new_rows["timestamp"] = new_rows["timestamp"].dt.strftime(TS_FMT)

    rows = []
    for r in new_rows.itertuples(index=False):
        if pd.isna(r.num_vehicles_available):
            continue  # toujours pas d'info → on saute
        rows.append((
            r.timestamp, int(r.station_index), r.station_name,
            float(r.lat) if pd.notna(r.lat) else None,
            float(r.lon) if pd.notna(r.lon) else None,
            int(r.capacity) if pd.notna(r.capacity) else 0,
            int(round(r.num_docks_available)) if pd.notna(r.num_docks_available) else None,
            int(round(r.num_docks_disabled)) if pd.notna(r.num_docks_disabled) else None,
            int(round(r.num_vehicles_available)),
            int(round(r.num_vehicles_disabled)) if pd.notna(r.num_vehicles_disabled) else None,
            int(r.count_x2) if pd.notna(r.count_x2) else 0,
        ))

    cur = conn.cursor()
    cur.executemany(INSERT_SQL, rows)
    return len(rows)


def main():
    with pipeline_step("fill_gaps"):
        conn = connect()
        try:
            n = fill(conn)
            conn.commit()
        finally:
            conn.close()
        print(f"OK fill_gaps — {n} lignes imputées insérées")


if __name__ == "__main__":
    main()
