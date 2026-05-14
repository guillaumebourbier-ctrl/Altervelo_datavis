#!/usr/bin/env python3
"""Crée les tables d'ingestion (raw_* + clean_*) dans velos.db.

Idempotent : `CREATE TABLE IF NOT EXISTS`. À lancer une fois avant le premier tick.
Les tables existantes du dashboard (stations, observations, predictions, pipeline_runs)
sont laissées intactes — elles sont créées par db_init.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from db import connect

SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_station_status (
    ts_collect TEXT NOT NULL,
    station_id TEXT NOT NULL,
    is_installed INTEGER,
    is_renting INTEGER,
    is_returning INTEGER,
    num_docks_available INTEGER,
    num_docks_disabled INTEGER,
    num_vehicles_available INTEGER,
    num_vehicles_disabled INTEGER,
    last_reported INTEGER,
    vehicle_types_available TEXT,
    PRIMARY KEY (ts_collect, station_id)
);
CREATE INDEX IF NOT EXISTS idx_rss_ts ON raw_station_status(ts_collect);

CREATE TABLE IF NOT EXISTS raw_vehicle_status (
    ts_collect TEXT NOT NULL,
    vehicle_id TEXT NOT NULL,
    vehicle_type_id TEXT,
    lat REAL,
    lon REAL,
    current_fuel_percent REAL,
    current_range_meters REAL,
    is_disabled INTEGER,
    is_reserved INTEGER,
    last_reported INTEGER,
    station_id TEXT,
    PRIMARY KEY (ts_collect, vehicle_id)
);
CREATE INDEX IF NOT EXISTS idx_rvs_ts ON raw_vehicle_status(ts_collect);

CREATE TABLE IF NOT EXISTS stations_clean (
    timestamp TEXT NOT NULL,
    station_index INTEGER NOT NULL,
    station_name TEXT,
    lat REAL,
    lon REAL,
    capacity INTEGER,
    num_docks_available INTEGER,
    num_docks_disabled INTEGER,
    num_vehicles_available INTEGER,
    num_vehicles_disabled INTEGER,
    count_x2 INTEGER,
    is_imputed INTEGER,
    PRIMARY KEY (timestamp, station_index)
);
CREATE INDEX IF NOT EXISTS idx_sc_ts ON stations_clean(timestamp);

CREATE TABLE IF NOT EXISTS vehicles_clean (
    timestamp TEXT NOT NULL,
    vehicle_id TEXT NOT NULL,
    station_index INTEGER NOT NULL,
    lat REAL,
    lon REAL,
    current_fuel_percent REAL,
    current_range_meters REAL,
    is_disabled INTEGER,
    PRIMARY KEY (timestamp, vehicle_id)
);
CREATE INDEX IF NOT EXISTS idx_vc_ts ON vehicles_clean(timestamp);
"""


def main():
    conn = connect()
    conn.executescript(SCHEMA)
    conn.commit()
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()]
    conn.close()
    print(f"OK — tables présentes : {', '.join(tables)}")


if __name__ == "__main__":
    main()
