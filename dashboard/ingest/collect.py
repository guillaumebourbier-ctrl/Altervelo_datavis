#!/usr/bin/env python3
"""Étape 1 — Fetch GBFS APIs et persiste les snapshots bruts dans velos.db.

Remplace ../collect.py (CSV append-only) par INSERT OR IGNORE SQL.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from db import connect  # noqa: E402

from _common import pipeline_step  # noqa: E402

APIS = {
    "vehicle_status": "https://api.gbfs.v3.0.ecovelo.mobi/altervelo/vehicle_status.json",
    "station_status": "https://api.gbfs.v3.0.ecovelo.mobi/altervelo/station_status.json",
}

INSERT_STATION = """INSERT OR IGNORE INTO raw_station_status
    (ts_collect, station_id, is_installed, is_renting, is_returning,
     num_docks_available, num_docks_disabled,
     num_vehicles_available, num_vehicles_disabled,
     last_reported, vehicle_types_available)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""

INSERT_VEHICLE = """INSERT OR IGNORE INTO raw_vehicle_status
    (ts_collect, vehicle_id, vehicle_type_id, lat, lon,
     current_fuel_percent, current_range_meters,
     is_disabled, is_reserved, last_reported, station_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""


def fetch_json(url: str) -> dict:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def collect_stations(conn, ts: str) -> int:
    data = fetch_json(APIS["station_status"])["data"]["stations"]
    rows = [(
        ts, s.get("station_id"),
        s.get("is_installed"), s.get("is_renting"), s.get("is_returning"),
        s.get("num_docks_available"), s.get("num_docks_disabled"),
        s.get("num_vehicles_available"), s.get("num_vehicles_disabled"),
        s.get("last_reported"),
        json.dumps(s.get("vehicle_types_available", [])),
    ) for s in data]
    cur = conn.cursor()
    cur.executemany(INSERT_STATION, rows)
    return cur.rowcount


def collect_vehicles(conn, ts: str) -> int:
    data = fetch_json(APIS["vehicle_status"])["data"]["vehicles"]
    rows = [(
        ts, v.get("vehicle_id"), v.get("vehicle_type_id"),
        v.get("lat"), v.get("lon"),
        v.get("current_fuel_percent"), v.get("current_range_meters"),
        v.get("is_disabled"), v.get("is_reserved"),
        v.get("last_reported"),
        v.get("station_id") or "",
    ) for v in data]
    cur = conn.cursor()
    cur.executemany(INSERT_VEHICLE, rows)
    return cur.rowcount


def main():
    with pipeline_step("collect"):
        ts = datetime.now(timezone.utc).isoformat()
        conn = connect()
        try:
            n_s = collect_stations(conn, ts)
            n_v = collect_vehicles(conn, ts)
            conn.commit()
        finally:
            conn.close()
        print(f"OK collect — ts={ts}  stations:+{n_s}  vehicles:+{n_v}")


if __name__ == "__main__":
    main()
