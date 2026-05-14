#!/usr/bin/env python3
"""Calcule les features de flux inter-stations à partir de vehicles_clean.csv.

Pour chaque (station, timestamp), compte :
  - n_arrivees_{lo}_{hi}m_{w}min : vélos présents dans la bande [lo, hi]m
    autour de la station à t mais pas à t-w
  - n_departs_{lo}_{hi}m_{w}min  : symétrique

Bandes : 0-150 m, 150-300 m
Fenêtres : 30, 60, 120 min
=> 2 bandes × 3 fenêtres × 2 directions = 12 features par (station, t).

Hypothèse clé : `vehicle_id` est stable entre snapshots (vérifié sur l'historique :
chaque vélo apparaît en moyenne 918 fois sur 978 timestamps possibles).

Sortie : data/vehicle_flow.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Détection auto du layout : repo (script/ + data/ frères) vs. dossier plat (Colab).
_HERE = Path(__file__).resolve().parent
if (_HERE.parent / "data").exists() or _HERE.name == "script":
    ROOT = _HERE.parent
else:
    ROOT = _HERE

_VC_CANDIDATES = [ROOT / "data/vehicles_clean.csv", ROOT / "vehicles_clean.csv"]
_SC_CANDIDATES = [ROOT / "data/stations_clean.csv", ROOT / "stations_clean.csv"]
VC_PATH = next((p for p in _VC_CANDIDATES if p.exists()), _VC_CANDIDATES[0])
SC_PATH = next((p for p in _SC_CANDIDATES if p.exists()), _SC_CANDIDATES[0])
OUT_DIR = ROOT / "data" if (ROOT / "data").exists() else ROOT
OUT_PATH = OUT_DIR / "vehicle_flow.csv"

BANDS = [(0, 150), (150, 300)]              # bandes principales (3 fenêtres)
BANDS_WIDE = [(300, 450), (450, 600)]        # bandes larges (fenêtre 60 min seulement)
WINDOWS_MIN = [30, 60, 120]
WINDOWS_WIDE_MIN = [60]
ALL_BANDS = BANDS + BANDS_WIDE


def haversine_m(lat1, lon1, lat2, lon2):
    """Distance haversine en mètres. Vectorisé NumPy."""
    R = 6_371_000.0
    lat1r = np.radians(lat1)
    lat2r = np.radians(lat2)
    dlat = lat2r - lat1r
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def feature_columns() -> list[str]:
    cols = []
    # bandes principales × 3 fenêtres × 2 directions = 12
    for w in WINDOWS_MIN:
        for lo, hi in BANDS:
            cols.append(f"n_arrivees_{lo}_{hi}m_{w}min")
            cols.append(f"n_departs_{lo}_{hi}m_{w}min")
    # bandes larges × 1 fenêtre × 2 directions = 4
    for w in WINDOWS_WIDE_MIN:
        for lo, hi in BANDS_WIDE:
            cols.append(f"n_arrivees_{lo}_{hi}m_{w}min")
            cols.append(f"n_departs_{lo}_{hi}m_{w}min")
    # net flow (arrivees - departs) sur bandes principales × 3 fenêtres = 6
    for w in WINDOWS_MIN:
        for lo, hi in BANDS:
            cols.append(f"net_flow_{lo}_{hi}m_{w}min")
    return cols


def build_state(vehicles: pd.DataFrame, station_coords: pd.DataFrame) -> dict:
    """state[ts] = dict {(station_index, band_idx): frozenset(vehicle_ids)}.

    band_idx indexe ALL_BANDS = BANDS + BANDS_WIDE :
      0 = [0,150), 1 = [150,300), 2 = [300,450), 3 = [450,600).
    """
    s_idx = station_coords["station_index"].to_numpy()
    s_lat = station_coords["lat"].to_numpy()
    s_lon = station_coords["lon"].to_numpy()

    state: dict = {}
    for ts, grp in vehicles.groupby("timestamp"):
        v_lat = grp["lat"].to_numpy()
        v_lon = grp["lon"].to_numpy()
        v_id = grp["vehicle_id"].to_numpy()
        if len(v_lat) == 0:
            state[ts] = {(int(sid), b): frozenset()
                         for sid in s_idx for b in range(len(ALL_BANDS))}
            continue
        # matrice n_stations × n_vehicles
        d = haversine_m(s_lat[:, None], s_lon[:, None], v_lat[None, :], v_lon[None, :])
        band_idx = np.full(d.shape, -1, dtype=np.int8)
        for b, (lo, hi) in enumerate(ALL_BANDS):
            band_idx[(d >= lo) & (d < hi)] = b

        st_state = {}
        for k, sid in enumerate(s_idx):
            for b in range(len(ALL_BANDS)):
                vids = v_id[band_idx[k] == b]
                st_state[(int(sid), b)] = frozenset(vids.tolist())
        state[ts] = st_state
    return state


def compute_flow(state: dict, station_indices: np.ndarray) -> pd.DataFrame:
    """Pour chaque (timestamp, station), calcule les 22 features de flux :
       12 (bandes principales × 3 fenêtres × 2 dir)
     +  4 (bandes larges × 1 fenêtre × 2 dir)
     +  6 (net_flow sur bandes principales × 3 fenêtres).

    Recherche stricte du tick t-w (égalité timestamp). Si absent, NaN.
    Le `dropna(subset=needed)` côté training éliminera les lignes critiques —
    XGBoost gère nativement les NaN sur les features non-essentielles.
    """
    timestamps = sorted(state.keys())
    rows = []
    for t in timestamps:
        for sid in station_indices:
            sid_int = int(sid)
            cells = {"timestamp": t, "station_index": sid_int}

            # Bandes principales : 3 fenêtres × 2 directions + net flow
            for b, (lo, hi) in enumerate(BANDS):
                now = state[t][(sid_int, b)]
                for w in WINDOWS_MIN:
                    target = t - pd.Timedelta(minutes=w)
                    prev_state = state.get(target)
                    if prev_state is None:
                        n_arr = n_dep = np.nan
                    else:
                        prev = prev_state[(sid_int, b)]
                        n_arr = len(now - prev)
                        n_dep = len(prev - now)
                    cells[f"n_arrivees_{lo}_{hi}m_{w}min"] = n_arr
                    cells[f"n_departs_{lo}_{hi}m_{w}min"] = n_dep
                    cells[f"net_flow_{lo}_{hi}m_{w}min"] = (
                        np.nan if prev_state is None else n_arr - n_dep
                    )

            # Bandes larges : seulement fenêtre 60 min, pas de net flow
            for b_offset, (lo, hi) in enumerate(BANDS_WIDE):
                b = len(BANDS) + b_offset
                now = state[t][(sid_int, b)]
                for w in WINDOWS_WIDE_MIN:
                    target = t - pd.Timedelta(minutes=w)
                    prev_state = state.get(target)
                    if prev_state is None:
                        cells[f"n_arrivees_{lo}_{hi}m_{w}min"] = np.nan
                        cells[f"n_departs_{lo}_{hi}m_{w}min"] = np.nan
                    else:
                        prev = prev_state[(sid_int, b)]
                        cells[f"n_arrivees_{lo}_{hi}m_{w}min"] = len(now - prev)
                        cells[f"n_departs_{lo}_{hi}m_{w}min"] = len(prev - now)
            rows.append(cells)
    return pd.DataFrame(rows)


def main():
    print(f"1/3 lecture {VC_PATH} et {SC_PATH}…")
    vehicles = pd.read_csv(VC_PATH, parse_dates=["timestamp"])
    stations = pd.read_csv(SC_PATH, parse_dates=["timestamp"])
    print(f"   vehicles : {len(vehicles):,} lignes  ;  "
          f"timestamps={vehicles.timestamp.nunique()}  ;  vélos={vehicles.vehicle_id.nunique()}")

    station_coords = (stations[["station_index", "lat", "lon"]]
                      .drop_duplicates("station_index")
                      .sort_values("station_index")
                      .reset_index(drop=True))
    print(f"   stations : {len(station_coords)} positions distinctes")

    print("2/3 calcul des appartenances aux bandes par tick…")
    state = build_state(vehicles, station_coords)

    print("3/3 calcul des flux (arrivées/départs) sur 30/60/120 min…")
    s_idx = station_coords["station_index"].to_numpy()
    flow = compute_flow(state, s_idx)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    flow.to_csv(OUT_PATH, index=False)
    print(f"OK -> {OUT_PATH} ({len(flow):,} lignes × {flow.shape[1]} colonnes)")

    # Sanity : combien de lignes avec au moins 1 arrivée ou départ ?
    feat_cols = feature_columns()
    n_total = len(flow)
    n_with_signal = (flow[feat_cols].fillna(0).sum(axis=1) > 0).sum()
    n_with_nan = flow[feat_cols].isna().any(axis=1).sum()
    print(f"   lignes avec ≥1 mouvement : {n_with_signal:,} ({n_with_signal/n_total*100:.1f}%)")
    print(f"   lignes avec ≥1 NaN (gap dans la grille temporelle) : {n_with_nan:,} "
          f"({n_with_nan/n_total*100:.1f}%)")


if __name__ == "__main__":
    main()
