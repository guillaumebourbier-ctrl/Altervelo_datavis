#!/usr/bin/env python3
"""Comble les trous de stations.csv sur une grille 30 min.

Stratégie :
- Nuit (20h-5h UTC+4) : forward-fill.
- Jour (5h-20h UTC+4) : interpolation linéaire des compteurs numériques,
  forward-fill pour les booléens / textes.
- Colonne is_imputed marque les lignes synthétiques.
"""

import csv
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/stations.csv"
DST = ROOT / "data/stations_filled.csv"

GRID_MIN = 30
TZ_REUNION = timezone(timedelta(hours=4))

NUMERIC_COLS = [
    "num_docks_available", "num_docks_disabled",
    "num_vehicles_available", "num_vehicles_disabled",
]
FFILL_COLS = [
    "is_installed", "is_renting", "is_returning",
    "last_reported", "vehicle_types_available",
]
OUT_FIELDS = [
    "timestamp", "station_index", "station_name",
    *FFILL_COLS[:3], *NUMERIC_COLS, *FFILL_COLS[3:], "is_imputed",
]


def round_to_grid(dt: datetime) -> datetime:
    """Arrondit dt au slot 30 min le plus proche (UTC)."""
    base = dt.replace(minute=0, second=0, microsecond=0)
    minutes = (dt - base).total_seconds() / 60
    slot = round(minutes / GRID_MIN) * GRID_MIN
    return base + timedelta(minutes=slot)


def is_night(slot_utc: datetime) -> bool:
    """Slot situé entre 20h et 5h heure Réunion (UTC+4)."""
    h = slot_utc.astimezone(TZ_REUNION).hour
    return h >= 20 or h < 5


def main():
    # 1. Charger : par station -> {slot -> row}
    by_station: dict[tuple[int, str], dict[datetime, dict]] = defaultdict(dict)
    all_slots = set()
    with SRC.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ts = datetime.fromisoformat(row["timestamp"])
            slot = round_to_grid(ts)
            key = (int(row["station_index"]), row["station_name"])
            # Garder la mesure dont ts est la plus proche du centre du slot
            existing = by_station[key].get(slot)
            if existing is None or abs(ts - slot) < abs(existing["_ts"] - slot):
                row["_ts"] = ts
                by_station[key][slot] = row
            all_slots.add(slot)

    if not all_slots:
        print("Aucune donnée.")
        return

    grid = []
    cur, end = min(all_slots), max(all_slots)
    while cur <= end:
        grid.append(cur)
        cur += timedelta(minutes=GRID_MIN)
    print(f"Grille : {len(grid)} slots de {grid[0]} à {grid[-1]}")

    # 2. Pour chaque station, construire la série complète
    rows_out = []
    nb_imputed = 0
    for (idx, name), measured in sorted(by_station.items()):
        sorted_slots = sorted(measured.keys())

        for slot in grid:
            if slot in measured:
                src = measured[slot]
                rows_out.append(_emit(slot, idx, name, src, imputed=False))
                continue

            # Trouver prev (dernière mesure < slot) et nxt (première > slot)
            prev_slot = max((s for s in sorted_slots if s < slot), default=None)
            next_slot = min((s for s in sorted_slots if s > slot), default=None)
            prev = measured.get(prev_slot) if prev_slot else None
            nxt = measured.get(next_slot) if next_slot else None

            if prev is None and nxt is None:
                continue  # station sans donnée
            if prev is None:
                synth = dict(nxt)
            elif nxt is None or is_night(slot):
                synth = dict(prev)
            else:
                # Jour : interpolation linéaire des numériques, ffill pour le reste
                synth = dict(prev)
                total = (next_slot - prev_slot).total_seconds()
                frac = (slot - prev_slot).total_seconds() / total
                for col in NUMERIC_COLS:
                    try:
                        a = float(prev[col]); b = float(nxt[col])
                        synth[col] = str(round(a + frac * (b - a)))
                    except (TypeError, ValueError):
                        pass

            rows_out.append(_emit(slot, idx, name, synth, imputed=True))
            nb_imputed += 1

    with DST.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Écrit {DST} : {len(rows_out)} lignes ({nb_imputed} imputées, "
          f"{len(rows_out) - nb_imputed} mesurées) sur {len(by_station)} stations.")


def _emit(slot: datetime, idx: int, name: str, src: dict, imputed: bool) -> dict:
    out = {
        "timestamp": slot.isoformat(),
        "station_index": idx,
        "station_name": name,
        "is_imputed": "True" if imputed else "False",
    }
    for col in NUMERIC_COLS + FFILL_COLS:
        out[col] = src.get(col, "")
    return out


if __name__ == "__main__":
    main()
