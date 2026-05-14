#!/usr/bin/env python3
"""Pipeline de nettoyage de vehicle_status.csv (sortie de collect.py).

Calqué sur cleaning_data.py mais pour la granularité vélo (et non station).

Étapes :
  1. lecture brute
  2. mapping station_id (hash) -> station_index (entier ; 0 = en circulation)
  3. quantification temporelle au pas 30 min (closest-to-center si doublons)
  4. localisation UTC -> UTC+4 (Réunion)
  5. diagnostic + drop des colonnes constantes (>90% valeur dominante)
  6. casts et bornes

Entrée  : data/vehicle_status.csv
Sortie  : data/vehicles_clean.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cleaning_data import fetch_station_mapping, TZ_REUNION, GRID_MIN

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SRC = DATA / "vehicle_status.csv"
DST = DATA / "vehicles_clean.csv"
STATIONS = DATA / "stations_clean.csv"  # pour le recadrage temporel

CONSTANT_THRESHOLD = 0.90  # part de la valeur dominante au-dessus de laquelle on drop


# ============================================================
# 1-2. CHARGEMENT + MAPPING station_id -> station_index
# ============================================================
def load_and_map() -> pd.DataFrame:
    df = pd.read_csv(SRC)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    mapping = fetch_station_mapping()  # {station_id: (index, name)}
    sid_to_index = {sid: idx for sid, (idx, _) in mapping.items()}

    # station_id vide / NaN -> 0 (vélo en circulation)
    sid = df["station_id"].fillna("")
    df["station_index"] = sid.map(sid_to_index).fillna(0).astype(int)

    # Vérifier qu'aucun station_id non-vide n'a échappé au mapping
    unknown = df.loc[(sid != "") & (df["station_index"] == 0), "station_id"].unique()
    if len(unknown):
        raise ValueError(f"{len(unknown)} station_id inconnus du mapping : {unknown[:3]}…")

    return df.drop(columns=["station_id"])


# ============================================================
# 3. QUANTIFICATION 30 MIN (closest-to-center par vélo)
# ============================================================
def quantize_30min(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["slot"] = df["timestamp"].dt.round(f"{GRID_MIN}min")
    df["dist"] = (df["timestamp"] - df["slot"]).abs()

    n_before = len(df)
    df = (df.sort_values("dist")
            .drop_duplicates(["vehicle_id", "slot"], keep="first")
            .drop(columns=["dist", "timestamp"])
            .rename(columns={"slot": "timestamp"}))
    n_dup = n_before - len(df)
    print(f"   doublons (vehicle_id, slot) écrasés : {n_dup:,}")
    return df


# ============================================================
# 4. LOCALISATION UTC -> UTC+4
# ============================================================
def localize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = df["timestamp"].dt.tz_convert(TZ_REUNION)
    return df


# ============================================================
# 5. DIAGNOSTIC + DROP DES CONSTANTES
# ============================================================
def drop_constant_cols(df: pd.DataFrame) -> pd.DataFrame:
    # Colonnes structurelles à ne JAMAIS dropper même si elles paraissent constantes
    protected = {"timestamp", "vehicle_id", "station_index"}

    to_drop = []
    print("   diagnostic colonnes :")
    for col in df.columns:
        if col in protected:
            continue
        top_share = df[col].value_counts(dropna=False, normalize=True).iloc[0]
        flag = "DROP" if top_share > CONSTANT_THRESHOLD else "    "
        print(f"     [{flag}] {col:<25s}  valeur dominante = {top_share:.1%}")
        if top_share > CONSTANT_THRESHOLD:
            to_drop.append(col)
    return df.drop(columns=to_drop)


# ============================================================
# 6a. FILTRE FANTÔMES
# ============================================================
def drop_phantom_vehicles(df: pd.DataFrame) -> pd.DataFrame:
    """Un vélo dont la batterie reste à 0% sur TOUTE la fenêtre d'observation
    est presque certainement un fantôme (vélo perdu/cassé que l'API continue
    de remonter). Un vélo réel est rechargé au moins une fois.
    """
    max_fuel = df.groupby("vehicle_id")["current_fuel_percent"].max()
    phantoms = max_fuel[max_fuel == 0.0].index
    n_lines = (df["vehicle_id"].isin(phantoms)).sum()
    print(f"   fantômes détectés : {len(phantoms)} vélos / {n_lines:,} lignes")
    return df[~df["vehicle_id"].isin(phantoms)].reset_index(drop=True)


# ============================================================
# 6b. RECADRAGE TEMPOREL SUR stations_clean
# ============================================================
def clip_to_stations_window(df: pd.DataFrame) -> pd.DataFrame:
    if not STATIONS.exists():
        print(f"   {STATIONS} absent -> pas de recadrage")
        return df
    s = pd.read_csv(STATIONS, parse_dates=["timestamp"], usecols=["timestamp"])
    tmin, tmax = s["timestamp"].min(), s["timestamp"].max()
    n_before = len(df)
    df = df[df["timestamp"].between(tmin, tmax)].reset_index(drop=True)
    print(f"   recadrage [{tmin} -> {tmax}] : {n_before - len(df):,} lignes hors fenêtre droppées")
    return df


# ============================================================
# 7. CASTS + BORNES
# ============================================================
def final_cast(df: pd.DataFrame) -> pd.DataFrame:
    if "is_disabled" in df.columns:
        df["is_disabled"] = df["is_disabled"].astype("int8")
    if "is_reserved" in df.columns:
        df["is_reserved"] = df["is_reserved"].astype("int8")
    if "current_fuel_percent" in df.columns:
        df["current_fuel_percent"] = df["current_fuel_percent"].clip(0.0, 1.0)
    return df


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"1/7 lecture {SRC} + mapping station_id…")
    df = load_and_map()
    print(f"   {len(df):,} lignes brutes ; {(df['station_index'] == 0).sum():,} en circulation")

    print("2/7 quantification 30 min…")
    df = quantize_30min(df)

    print("3/7 localisation UTC+4…")
    df = localize(df)

    print("4/7 drop des colonnes constantes…")
    df = drop_constant_cols(df)

    print("5/7 filtre fantômes (batterie toujours 0%)…")
    df = drop_phantom_vehicles(df)

    print("6/7 recadrage sur la fenêtre stations_clean…")
    df = clip_to_stations_window(df)

    print("7/7 casts finaux…")
    df = final_cast(df)

    # Réordonner pour lisibilité
    head = ["timestamp", "vehicle_id", "station_index"]
    cols = head + [c for c in df.columns if c not in head]
    df = df[cols].sort_values(["vehicle_id", "timestamp"]).reset_index(drop=True)

    DST.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DST, index=False)
    print(f"OK -> {DST} ({len(df):,} lignes × {df.shape[1]} colonnes)")


if __name__ == "__main__":
    main()
