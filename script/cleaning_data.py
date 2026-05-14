#!/usr/bin/env python3
"""Pipeline complet de nettoyage entre collect.py et l'entraînement.

Consolide en un seul script les 4 anciennes étapes :
  1. enrich_stations       : station_id opaque -> station_index + station_name
  2. fill_gaps             : grille 30 min, ffill la nuit, interpolation le jour
  3. localize_timestamps   : UTC -> UTC+4 (Réunion)
  4. final_cleaning_script : drop colonnes constantes, parse vehicle_types

Entrée  : data/station_status.csv (sortie de collect.py)
Sortie  : data/stations_clean.csv  (entrée des scripts d'entraînement)

Dépendances : pandas, requests
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ============================================================
# CONFIG
# ============================================================
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SRC = DATA / "station_status.csv"
DST = DATA / "stations_clean.csv"

INFO_URL = "https://api.gbfs.v3.0.ecovelo.mobi/altervelo/station_information.json"

GRID_MIN = 30                              # granularité cible
TZ_REUNION = "Etc/GMT-4"                   # UTC+4 (signe inversé en POSIX)
NIGHT_HOURS = set(list(range(20, 24)) + list(range(0, 5)))  # 20h-5h locale

NUMERIC_COLS = [
    "num_docks_available", "num_docks_disabled",
    "num_vehicles_available", "num_vehicles_disabled",
]


# ============================================================
# 1. ENRICHISSEMENT (station_id -> index + name)
# ============================================================
def fetch_station_info() -> pd.DataFrame:
    """Retourne un DataFrame avec station_id, station_index, station_name, lat, lon, capacity.

    Source unique pour toutes les métadonnées station — utilisée par cleaning_data
    et par clean_vehicle_status (via fetch_station_mapping ci-dessous).
    """
    stations = requests.get(INFO_URL, timeout=30).json()["data"]["stations"]
    sorted_st = sorted(stations, key=lambda s: s["station_id"])
    return pd.DataFrame([
        {
            "station_id": s["station_id"],
            "station_index": i + 1,
            "station_name": s["name"][0]["text"],
            "lat": s["lat"],
            "lon": s["lon"],
            "capacity": s.get("capacity", 0),
        }
        for i, s in enumerate(sorted_st)
    ])


def fetch_station_mapping() -> dict[str, tuple[int, str]]:
    """Wrapper rétro-compatible : {station_id: (index, name)}."""
    info = fetch_station_info()
    return {r.station_id: (r.station_index, r.station_name) for r in info.itertuples()}


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    info = fetch_station_info()
    info_map = {r.station_id: r for r in info.itertuples()}
    next_idx = info["station_index"].max() + 1
    extra_rows = []
    for sid in df["station_id"].unique():
        if sid not in info_map:
            extra_rows.append({"station_id": sid, "station_index": next_idx,
                               "station_name": "", "lat": np.nan, "lon": np.nan, "capacity": 0})
            next_idx += 1
    if extra_rows:
        info = pd.concat([info, pd.DataFrame(extra_rows)], ignore_index=True)
    df = df.merge(info[["station_id", "station_index", "station_name", "lat", "lon", "capacity"]],
                  on="station_id", how="left")
    return df.drop(columns=["station_id"])


# ============================================================
# 2. COMBLEMENT DES TROUS  (grille 30 min)
# ============================================================
def fill_gaps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    # Arrondir au slot 30 min le plus proche
    df["slot"] = df["timestamp"].dt.round(f"{GRID_MIN}min")
    # Une seule ligne par (station, slot) : la mesure la plus proche du centre
    df["dist"] = (df["timestamp"] - df["slot"]).abs()
    df = df.sort_values("dist").drop_duplicates(["station_index", "slot"], keep="first")
    df = df.drop(columns=["dist", "timestamp"]).rename(columns={"slot": "timestamp"})

    # Grille canonique (slot × station)
    slots = pd.date_range(df["timestamp"].min(), df["timestamp"].max(),
                          freq=f"{GRID_MIN}min", tz="UTC")
    stations = df[["station_index", "station_name", "lat", "lon", "capacity"]].drop_duplicates()
    grid = stations.merge(pd.DataFrame({"timestamp": slots}), how="cross")

    merged = grid.merge(df, on=["station_index", "station_name", "lat", "lon", "capacity", "timestamp"], how="left")
    merged["is_imputed"] = merged[NUMERIC_COLS[0]].isna()
    merged = merged.sort_values(["station_index", "timestamp"]).reset_index(drop=True)

    # Heure locale -> détection nuit
    local_hour = merged["timestamp"].dt.tz_convert(TZ_REUNION).dt.hour
    is_night = local_hour.isin(NIGHT_HOURS)

    # Forward-fill par station pour TOUTES les colonnes (booleans, JSON…)
    grouped = merged.groupby("station_index", group_keys=False)
    ffilled = grouped.ffill()

    # Interpolation linéaire de jour pour les compteurs numériques
    interp = grouped[NUMERIC_COLS].apply(lambda g: g.interpolate(method="linear", limit_direction="both"))
    for col in NUMERIC_COLS:
        merged[col] = np.where(is_night, ffilled[col], interp[col])
    # Reste : ffill puis bfill (premières lignes éventuelles)
    other_cols = [c for c in merged.columns if c not in NUMERIC_COLS + ["timestamp", "station_index", "station_name", "lat", "lon", "capacity", "is_imputed"]]
    merged[other_cols] = grouped[other_cols].ffill().bfill()
    return merged


# ============================================================
# 3. LOCALISATION UTC -> UTC+4
# ============================================================
def localize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = df["timestamp"].dt.tz_convert(TZ_REUNION)
    return df


# ============================================================
# 4. NETTOYAGE FINAL (drop + cast + parse JSON)
# ============================================================
def extract_x2(json_str) -> int:
    if not isinstance(json_str, str):
        return 0
    try:
        for vt in json.loads(json_str):
            if vt.get("vehicle_type_id") == "x2":
                return int(vt.get("count", 0))
    except Exception:
        pass
    return 0


def final_clean(df: pd.DataFrame) -> pd.DataFrame:
    # Drop colonnes constantes (variation < 10%) et last_reported
    for col in ["is_installed", "is_renting", "is_returning", "last_reported"]:
        if col in df.columns:
            df = df.drop(columns=col)

    df["count_x2"] = df["vehicle_types_available"].apply(extract_x2)
    df = df.drop(columns=["vehicle_types_available"])

    for c in NUMERIC_COLS:
        df[c] = df[c].round().astype(int)
    df["station_index"] = df["station_index"].astype(int)

    cols = ["timestamp", "station_index", "station_name", "lat", "lon", "capacity",
            *NUMERIC_COLS, "count_x2", "is_imputed"]
    return df[cols]


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"1/4 lecture {SRC}…")
    df = pd.read_csv(SRC)
    print(f"   {len(df):,} lignes brutes")

    print("2/4 enrichissement station_id -> index/name…")
    df = enrich(df)

    print("3/4 grille 30 min + ffill nuit / interp jour…")
    df = fill_gaps(df)
    n_imp = int(df["is_imputed"].sum())
    print(f"   {len(df):,} lignes ({n_imp:,} imputées)")

    print("4/4 localisation UTC+4 + nettoyage final…")
    df = localize(df)
    df = final_clean(df)

    DST.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DST, index=False)
    print(f"OK -> {DST} ({len(df):,} lignes × {df.shape[1]} colonnes)")


if __name__ == "__main__":
    main()
