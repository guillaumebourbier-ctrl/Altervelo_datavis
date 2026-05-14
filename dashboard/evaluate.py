#!/usr/bin/env python3
"""Backfill différé : pour chaque prédiction dont ts_target est passé,
remplit y_obs / err_model / err_pers depuis observations.

Idempotent. À lancer après chaque tick collect+predict.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

DASHBOARD_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DASHBOARD_DIR))

from db import backfill_predictions, connect


def main():
    t0 = time.time()
    conn = connect()
    cur = conn.cursor()
    n_before = cur.execute(
        "SELECT COUNT(*) FROM predictions WHERE y_obs IS NULL"
    ).fetchone()[0]

    n_total = backfill_predictions(conn)

    n_after = cur.execute(
        "SELECT COUNT(*) FROM predictions WHERE y_obs IS NULL"
    ).fetchone()[0]
    n_filled = n_before - n_after
    conn.close()
    print(f"OK : +{n_filled} prédictions évaluées (total avec y_obs : {n_total:,}) "
          f"en {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
