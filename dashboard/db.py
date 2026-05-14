"""Helpers SQLite partagés (connexion + backfill)."""
from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "velos.db"


def connect() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def backfill_predictions(conn: sqlite3.Connection) -> int:
    """Remplit y_obs / err_model / err_pers pour les prédictions dont l'observation
    cible est désormais disponible. Idempotent (UPDATE WHERE y_obs IS NULL)."""
    cur = conn.cursor()
    cur.execute("""
        UPDATE predictions
        SET y_obs = (
            SELECT o.num_vehicles_available
            FROM observations o
            WHERE o.timestamp = predictions.ts_target
              AND o.station_index = predictions.station_index
        )
        WHERE y_obs IS NULL
    """)
    cur.execute("""
        UPDATE predictions
        SET err_model = ABS(y_pred - y_obs),
            err_pers  = ABS(y_current - y_obs)
        WHERE y_obs IS NOT NULL AND err_model IS NULL
    """)
    conn.commit()
    n_total = cur.execute(
        "SELECT COUNT(*) FROM predictions WHERE y_obs IS NOT NULL"
    ).fetchone()[0]
    return n_total
