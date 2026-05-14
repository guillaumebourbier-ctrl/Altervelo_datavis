#!/usr/bin/env python3
"""Étape 4 — raw_vehicle_status → vehicles_clean.

Pour chaque vélo : map station_id (0 = transit), quantize 30 min closest-to-center,
localize UTC+4, drop colonnes constantes (>90 % dominance), filtre fantômes
(batterie toujours 0%), recadre sur la fenêtre stations_clean, casts.

Sortie : vehicles_clean (PK timestamp, vehicle_id).

Idempotent : INSERT OR REPLACE — un vélo réobservé sur le même slot écrase la
ligne précédente (les valeurs peuvent être affinées par closest-to-center).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from db import connect  # noqa: E402

from _common import GRID_MIN, TZ_REUNION, fetch_station_mapping, pipeline_step  # noqa: E402

CONSTANT_THRESHOLD = 0.90
TS_FMT = "%Y-%m-%dT%H:%M:%S%z"

# Schéma fixe stables après drop des colonnes constantes (cf. ../clean_vehicle_status.py).
# Ces colonnes sont les seules persistées dans vehicles_clean.
KEPT_COLS = ["lat", "lon", "current_fuel_percent", "current_range_meters", "is_disabled"]

INSERT_SQL = """INSERT OR REPLACE INTO vehicles_clean
    (timestamp, vehicle_id, station_index,
     lat, lon, current_fuel_percent, current_range_meters, is_disabled)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)"""


def clean(conn) -> int:
    df = pd.read_sql("SELECT * FROM raw_vehicle_status", conn,
                     parse_dates=["ts_collect"])
    if df.empty:
        return 0

    # Mapping station_id -> station_index (0 = transit)
    mapping = fetch_station_mapping()
    sid_to_idx = {sid: idx for sid, (idx, _) in mapping.items()}
    sid = df["station_id"].fillna("")
    df["station_index"] = sid.map(sid_to_idx).fillna(0).astype(int)

    # Quantize 30-min closest-to-center par (vehicle_id, slot)
    df["ts_collect"] = pd.to_datetime(df["ts_collect"], utc=True)
    df["slot"] = df["ts_collect"].dt.round(f"{GRID_MIN}min")
    df["dist"] = (df["ts_collect"] - df["slot"]).abs()
    df = (df.sort_values("dist")
            .drop_duplicates(["vehicle_id", "slot"], keep="first")
            .drop(columns=["dist", "ts_collect", "station_id"])
            .rename(columns={"slot": "timestamp"}))

    # Localize UTC+4
    df["timestamp"] = df["timestamp"].dt.tz_convert(TZ_REUNION)

    # Filtre fantômes : current_fuel_percent toujours 0 sur toute la fenêtre
    if "current_fuel_percent" in df.columns:
        max_fuel = df.groupby("vehicle_id")["current_fuel_percent"].max()
        phantoms = max_fuel[max_fuel == 0.0].index
        df = df[~df["vehicle_id"].isin(phantoms)].reset_index(drop=True)

    # Recadrage sur la fenêtre stations_clean
    sc = pd.read_sql("SELECT MIN(timestamp) AS tmin, MAX(timestamp) AS tmax "
                     "FROM stations_clean", conn)
    if not sc.empty and pd.notna(sc.loc[0, "tmin"]):
        tmin = pd.to_datetime(sc.loc[0, "tmin"], utc=True).tz_convert(TZ_REUNION)
        tmax = pd.to_datetime(sc.loc[0, "tmax"], utc=True).tz_convert(TZ_REUNION)
        df = df[df["timestamp"].between(tmin, tmax)].reset_index(drop=True)

    if df.empty:
        return 0

    # Casts + clip batterie
    if "is_disabled" in df.columns:
        df["is_disabled"] = pd.to_numeric(df["is_disabled"], errors="coerce") \
                              .fillna(0).astype("int8")
    if "current_fuel_percent" in df.columns:
        df["current_fuel_percent"] = df["current_fuel_percent"].clip(0.0, 1.0)

    df["timestamp"] = df["timestamp"].dt.strftime(TS_FMT)

    rows = [
        (
            r.timestamp, str(r.vehicle_id), int(r.station_index),
            float(r.lat) if pd.notna(r.lat) else None,
            float(r.lon) if pd.notna(r.lon) else None,
            float(r.current_fuel_percent) if pd.notna(r.current_fuel_percent) else None,
            float(r.current_range_meters) if pd.notna(r.current_range_meters) else None,
            int(r.is_disabled) if pd.notna(r.is_disabled) else 0,
        )
        for r in df.itertuples(index=False)
    ]
    cur = conn.cursor()
    cur.executemany(INSERT_SQL, rows)
    return len(rows)


def main():
    with pipeline_step("clean_vehicles"):
        conn = connect()
        try:
            n = clean(conn)
            conn.commit()
        finally:
            conn.close()
        print(f"OK clean_vehicles — {n} lignes (INSERT OR REPLACE)")


if __name__ == "__main__":
    main()
