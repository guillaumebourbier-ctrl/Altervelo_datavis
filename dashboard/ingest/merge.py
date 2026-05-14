#!/usr/bin/env python3
"""Étape 5 — stations_clean + vehicles_clean → observations.

Reproduit ../merge_csv.py côté DB :
  - agrégats vélos « à la borne » (station_index ≠ 0)
  - histogramme spatial 6 bandes 150 m + dist_nearest_transit_m (vélos transit)
  - left-join sur stations_clean, ajoute is_obs_missing
  - imputation : NaN battery préservé, 0 sur compteurs, 10 000 sur dist
  - INSERT OR IGNORE dans observations avec source='live'

Le tuple OBS_COLS doit rester aligné avec append_live_obs.py — c'est la source
de vérité du contrat avec le modèle XGBoost (cf. CLAUDE.md train/serve invariant).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from db import connect  # noqa: E402

from _common import pipeline_step  # noqa: E402

LOW_BATTERY_THRESHOLD = 0.20
TRANSIT_BAND_M = 150
TRANSIT_BAND_COUNT = 6  # 0 → 900 m

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


def _band_label(i: int) -> str:
    return f"n_transit_{i * TRANSIT_BAND_M}_{(i + 1) * TRANSIT_BAND_M}m"


BAND_COLS = [_band_label(i) for i in range(TRANSIT_BAND_COUNT)]


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000.0
    lat1r = np.radians(lat1); lat2r = np.radians(lat2)
    dlat = lat2r - lat1r
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def aggregate_at_dock(vehicles: pd.DataFrame) -> pd.DataFrame:
    docked = vehicles[vehicles["station_index"] != 0].copy()
    if docked.empty:
        return pd.DataFrame(columns=[
            "timestamp", "station_index", "n_vehicles_actifs",
            "n_vehicles_disabled_obs", "mean_battery", "min_battery",
            "max_battery", "pct_low_battery",
        ])
    docked["is_active"] = (docked["is_disabled"] == 0).astype(int)
    docked["is_low_batt"] = (docked["current_fuel_percent"] < LOW_BATTERY_THRESHOLD).astype(int)
    g = docked.groupby(["timestamp", "station_index"])
    return g.agg(
        n_vehicles_actifs=("is_active", "sum"),
        n_vehicles_disabled_obs=("is_disabled", "sum"),
        mean_battery=("current_fuel_percent", "mean"),
        min_battery=("current_fuel_percent", "min"),
        max_battery=("current_fuel_percent", "max"),
        pct_low_battery=("is_low_batt", "mean"),
    ).reset_index()


def aggregate_in_transit(stations: pd.DataFrame, vehicles: pd.DataFrame) -> pd.DataFrame:
    transit = vehicles[vehicles["station_index"] == 0].copy()
    coords = (stations[["station_index", "lat", "lon"]]
              .drop_duplicates("station_index")
              .sort_values("station_index")
              .reset_index(drop=True))
    s_idx = coords["station_index"].to_numpy()
    s_lat = coords["lat"].to_numpy()
    s_lon = coords["lon"].to_numpy()
    edges = np.array([(i + 1) * TRANSIT_BAND_M for i in range(TRANSIT_BAND_COUNT)])

    rows = []
    timestamps = sorted(stations["timestamp"].unique())
    transit_by_ts = dict(tuple(transit.groupby("timestamp")))
    for ts in timestamps:
        group = transit_by_ts.get(ts)
        if group is None or group.empty:
            for si in s_idx:
                rows.append((ts, si, *([0] * TRANSIT_BAND_COUNT), np.nan))
            continue
        v_lat = group["lat"].to_numpy()
        v_lon = group["lon"].to_numpy()
        d = haversine_m(s_lat[:, None], s_lon[:, None], v_lat[None, :], v_lon[None, :])
        cum = (d[:, :, None] <= edges[None, None, :]).sum(axis=1)
        bands = np.diff(cum, axis=1, prepend=0)
        d_min = d.min(axis=1)
        for k in range(len(s_idx)):
            rows.append((ts, s_idx[k], *[int(x) for x in bands[k]], float(d_min[k])))

    return pd.DataFrame(rows, columns=[
        "timestamp", "station_index", *BAND_COLS, "dist_nearest_transit_m",
    ])


def build_observations(conn) -> pd.DataFrame:
    stations = pd.read_sql("SELECT * FROM stations_clean", conn)
    vehicles = pd.read_sql("SELECT * FROM vehicles_clean", conn)
    if stations.empty:
        return pd.DataFrame(columns=OBS_COLS)

    dock_agg = aggregate_at_dock(vehicles)
    transit_agg = aggregate_in_transit(stations, vehicles)

    out = stations.merge(dock_agg, on=["timestamp", "station_index"], how="left")
    out = out.merge(transit_agg, on=["timestamp", "station_index"], how="left")

    out["is_obs_missing"] = out["n_vehicles_actifs"].isna().astype(int)
    counter_cols = ["n_vehicles_actifs", "n_vehicles_disabled_obs", *BAND_COLS]
    for c in counter_cols:
        out[c] = out[c].fillna(0).astype(int)
    out["dist_nearest_transit_m"] = out["dist_nearest_transit_m"].fillna(10_000.0)
    # mean/min/max_battery + pct_low_battery : NaN volontairement préservés
    return out[OBS_COLS]


def main():
    with pipeline_step("merge") as counters:
        conn = connect()
        try:
            df = build_observations(conn)
            if df.empty:
                print("OK merge — rien à insérer (stations_clean vide)")
                return
            rows = []
            for r in df.itertuples(index=False, name=None):
                # NaN -> None pour SQLite
                rows.append(tuple(None if (isinstance(v, float) and pd.isna(v)) else v
                                  for v in r))
            cur = conn.cursor()
            cur.executemany(INSERT_SQL, rows)
            conn.commit()
            n_new = cur.rowcount
            counters["n_obs_inserted"] = n_new
        finally:
            conn.close()
        print(f"OK merge — +{n_new} obs (source='live')")


if __name__ == "__main__":
    main()
