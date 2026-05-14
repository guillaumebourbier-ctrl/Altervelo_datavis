#!/usr/bin/env python3
"""Fusion stations_clean + vehicles_clean -> stations_enriched.csv.

Granularité de sortie identique à stations_clean : une ligne par (timestamp, station_index).
Les colonnes ajoutées proviennent de l'agrégation de vehicles_clean au même timestamp :
  - features "à la borne"   : agrégats sur les vélos dont station_index == s
  - features "en transit"   : haversine entre la station s et les vélos avec station_index == 0

Entrée  : data/stations_clean.csv, data/vehicles_clean.csv
Sortie  : data/stations_enriched.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SRC_STATIONS = DATA / "stations_clean.csv"
SRC_VEHICLES = DATA / "vehicles_clean.csv"
DST = DATA / "stations_enriched.csv"

LOW_BATTERY_THRESHOLD = 0.20
# Histogramme spatial des vélos en transit : 6 bandes concentriques de 150 m de large.
# Pas choisi à 150 m comme compromis entre granularité (capter les arrivées imminentes)
# et stabilité (chaque bande doit avoir un comptage non nul assez souvent pour être utile).
TRANSIT_BAND_M = 150
TRANSIT_BAND_COUNT = 6  # couvre 0 -> 900 m


# ============================================================
# 1. AGRÉGATS PAR (timestamp, station_index)  -- vélos "à la borne"
# ============================================================
def aggregate_at_dock(vehicles: pd.DataFrame) -> pd.DataFrame:
    """Pour chaque (t, s), résume les vélos présents à la borne s au temps t."""
    docked = vehicles[vehicles["station_index"] != 0].copy()
    docked["is_active"] = (docked["is_disabled"] == 0).astype(int)
    docked["is_low_batt"] = (docked["current_fuel_percent"] < LOW_BATTERY_THRESHOLD).astype(int)

    g = docked.groupby(["timestamp", "station_index"])
    agg = g.agg(
        n_vehicles_actifs=("is_active", "sum"),
        n_vehicles_disabled_obs=("is_disabled", "sum"),
        mean_battery=("current_fuel_percent", "mean"),
        min_battery=("current_fuel_percent", "min"),
        max_battery=("current_fuel_percent", "max"),
        pct_low_battery=("is_low_batt", "mean"),
    ).reset_index()
    return agg


# ============================================================
# 2. FEATURES SPATIALES  -- vélos "en transit"
# ============================================================
def haversine_m(lat1, lon1, lat2, lon2):
    """Distance haversine en mètres. lat/lon en degrés. Vectorisé numpy."""
    R = 6_371_000.0
    lat1r = np.radians(lat1); lat2r = np.radians(lat2)
    dlat = lat2r - lat1r
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def _band_label(i: int) -> str:
    lo, hi = i * TRANSIT_BAND_M, (i + 1) * TRANSIT_BAND_M
    return f"n_transit_{lo}_{hi}m"


def aggregate_in_transit(stations: pd.DataFrame, vehicles: pd.DataFrame) -> pd.DataFrame:
    """Pour chaque (t, s) : histogramme spatial (bandes 150 m) + distance au plus proche."""
    transit = vehicles[vehicles["station_index"] == 0].copy()
    station_coords = (stations[["station_index", "lat", "lon"]]
                      .drop_duplicates("station_index")
                      .sort_values("station_index")
                      .reset_index(drop=True))
    s_idx = station_coords["station_index"].to_numpy()
    s_lat = station_coords["lat"].to_numpy()
    s_lon = station_coords["lon"].to_numpy()
    n_stations = len(s_idx)

    band_cols = [_band_label(i) for i in range(TRANSIT_BAND_COUNT)]
    edges = np.array([(i + 1) * TRANSIT_BAND_M for i in range(TRANSIT_BAND_COUNT)])  # [150, 300, ...]

    rows = []
    for ts, group in transit.groupby("timestamp"):
        v_lat = group["lat"].to_numpy()
        v_lon = group["lon"].to_numpy()
        if len(v_lat) == 0:
            for si in s_idx:
                rows.append((ts, si, *([0] * TRANSIT_BAND_COUNT), np.nan))
            continue
        # matrice n_stations × n_transit
        d = haversine_m(s_lat[:, None], s_lon[:, None], v_lat[None, :], v_lon[None, :])
        # comptages cumulés ≤ edge[i] puis différences = comptages dans chaque bande
        cum = (d[:, :, None] <= edges[None, None, :]).sum(axis=1)  # n_stations × n_bands
        bands = np.diff(cum, axis=1, prepend=0)                    # n_stations × n_bands
        d_min = d.min(axis=1)
        for k in range(n_stations):
            rows.append((ts, s_idx[k], *[int(x) for x in bands[k]], float(d_min[k])))

    return pd.DataFrame(rows, columns=[
        "timestamp", "station_index",
        *band_cols, "dist_nearest_transit_m",
    ])


# ============================================================
# 3. FUSION + IMPUTATION
# ============================================================
def merge_all(stations: pd.DataFrame, dock_agg: pd.DataFrame, transit_agg: pd.DataFrame) -> pd.DataFrame:
    out = stations.merge(dock_agg, on=["timestamp", "station_index"], how="left")
    out = out.merge(transit_agg, on=["timestamp", "station_index"], how="left")

    # Marquer les paires (t, s) sans aucun vélo observé à la borne
    out["is_obs_missing"] = out["n_vehicles_actifs"].isna().astype("int8")

    # Compteurs : NaN -> 0 (pas d'observation == 0 vélo observé)
    band_cols = [_band_label(i) for i in range(TRANSIT_BAND_COUNT)]
    counter_cols = ["n_vehicles_actifs", "n_vehicles_disabled_obs", *band_cols]
    for c in counter_cols:
        out[c] = out[c].fillna(0).astype(int)

    # ============================================================
    # TODO(human) : stratégie d'imputation pour les batteries
    # ============================================================
    # Les colonnes mean_battery, min_battery, max_battery, pct_low_battery
    # sont NaN quand `is_obs_missing == 1` (aucun vélo observé à la borne au temps t).
    #
    # Trois options envisagées (cf. plan, partie B "Contribution Learn by Doing") :
    #   1. fillna(0) sur tout                       (simple, mais 0% batterie ≠ "pas d'info")
    #   2. fillna(0) compteurs, NaN sur batteries   (XGBoost gère nativement les NaN)
    #   3. valeur neutre + flag is_obs_missing      (déjà ajouté ci-dessus)
    #
    # Implémente la stratégie choisie ci-dessous (2-5 lignes attendues).
    # Le choix doit être cohérent avec le contrat du modèle aval (train_xgboostplus.py).
    # Choix : NaN sur les batteries quand aucun vélo n'est observé à la borne.
    # Justification : 0% est une valeur métier (vélo à plat), différente de "pas d'info".
    # XGBoost partitionne nativement les NaN ; combiné au flag is_obs_missing,
    # le modèle distingue proprement "non observé" de "observé bas".
    _ = ["mean_battery", "min_battery", "max_battery", "pct_low_battery"]  # NaN volontairement préservés

    # dist_nearest_transit_m : NaN = aucun vélo en transit dans le réseau au temps t.
    # Sentinelle large (10 km > diamètre du réseau) pour signaler "très loin".
    out["dist_nearest_transit_m"] = out["dist_nearest_transit_m"].fillna(10_000.0)
    return out


# ============================================================
# 4. SANITY CHECKS
# ============================================================
def sanity_check(out: pd.DataFrame) -> None:
    paired = out.dropna(subset=["n_vehicles_actifs"])
    corr = paired["n_vehicles_actifs"].corr(paired["num_vehicles_available"])
    diff = (paired["n_vehicles_actifs"] - paired["num_vehicles_available"]).abs()
    pct_le1 = (diff <= 1).mean() * 100
    pct_eq0 = (diff == 0).mean() * 100
    print(f"   corr(n_vehicles_actifs, num_vehicles_available) = {corr:.3f}  (attendu > 0.9)")
    print(f"   |diff| moyen = {diff.mean():.3f}  ;  |diff| ≤ 1 : {pct_le1:.1f}%  ;  diff = 0 : {pct_eq0:.1f}%")
    n_missing = int(out["is_obs_missing"].sum())
    print(f"   paires (t, s) sans observation vélo : {n_missing:,} / {len(out):,}  "
          f"({n_missing/len(out)*100:.1f}%)")


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"1/4 lecture {SRC_STATIONS} et {SRC_VEHICLES}…")
    stations = pd.read_csv(SRC_STATIONS, parse_dates=["timestamp"])
    vehicles = pd.read_csv(SRC_VEHICLES, parse_dates=["timestamp"])
    print(f"   stations : {len(stations):,} lignes  ;  vehicles : {len(vehicles):,} lignes")

    print("2/4 agrégation vélos « à la borne »…")
    dock_agg = aggregate_at_dock(vehicles)
    print(f"   {len(dock_agg):,} paires (t, s) agrégées")

    print("3/4 features spatiales (haversine vélos en transit)…")
    transit_agg = aggregate_in_transit(stations, vehicles)
    print(f"   {len(transit_agg):,} paires (t, s) avec features spatiales")

    print("4/4 fusion + imputation…")
    out = merge_all(stations, dock_agg, transit_agg)
    sanity_check(out)

    DST.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(DST, index=False)
    print(f"OK -> {DST} ({len(out):,} lignes × {out.shape[1]} colonnes)")


if __name__ == "__main__":
    main()
