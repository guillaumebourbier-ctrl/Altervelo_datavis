"""Helpers partagés par les scripts d'ingestion : station mapping, log run, contexte."""
from __future__ import annotations

import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from db import connect  # noqa: E402

INFO_URL = "https://api.gbfs.v3.0.ecovelo.mobi/altervelo/station_information.json"
TZ_REUNION = "Etc/GMT-4"
GRID_MIN = 30


def fetch_station_info() -> pd.DataFrame:
    """Récupère station_id, station_index, station_name, lat, lon, capacity.
    Source unique pour toutes les métadonnées station (calque de cleaning_data.py)."""
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
    info = fetch_station_info()
    return {r.station_id: (r.station_index, r.station_name) for r in info.itertuples()}


@contextmanager
def pipeline_step(name: str):
    """Logge la durée et le statut dans pipeline_runs (1 ligne par étape).

    Le name est juste informatif (préfixe du status), la PK ts_run est l'instant de fin.
    """
    t0 = time.time()
    counters = {"n_obs_inserted": 0, "n_pred_inserted": 0}
    try:
        yield counters
        status = f"ok:{name}"
        err = None
    except Exception as exc:
        status = f"error:{name}"
        err = f"{exc}\n{traceback.format_exc()}"
        raise
    finally:
        dur_ms = int((time.time() - t0) * 1000)
        ts_run = datetime.now(timezone.utc).isoformat()
        try:
            conn = connect()
            conn.execute(
                "INSERT OR REPLACE INTO pipeline_runs "
                "(ts_run, duration_ms, n_obs_inserted, n_pred_inserted, status, error_msg) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (ts_run, dur_ms, counters["n_obs_inserted"], counters["n_pred_inserted"],
                 status, err),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass  # ne jamais masquer l'erreur d'origine
