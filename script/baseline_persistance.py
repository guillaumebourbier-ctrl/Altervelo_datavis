#!/usr/bin/env python3
"""Calcule la MAE / RMSE de la baseline persistance sur stations_clean.csv.

Persistance = "je prédis que dans 1h la station aura le même nombre de vélos
qu'à l'instant courant". C'est la baseline naïve à battre.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data/stations_clean.csv"
HORIZON_STEPS = 2          # 2 × 30 min = prédire t+1h
TARGET_COL = "num_vehicles_available"
TEST_RATIO = 0.2

df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
df = df.sort_values(["station_index", "timestamp"]).reset_index(drop=True)

# y = valeur dans HORIZON_STEPS (cf. train_naive.py)
df["y"] = df.groupby("station_index")[TARGET_COL].shift(-HORIZON_STEPS)
df = df.dropna(subset=["y"])

# Split temporel identique aux scripts d'entraînement
cutoff = df["timestamp"].quantile(1 - TEST_RATIO)
test = df[df["timestamp"] >= cutoff]

y_true = test["y"].values
y_pred = test[TARGET_COL].values   # persistance pure

mae  = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print(f"Baseline persistance — horizon t+{HORIZON_STEPS*30} min")
print(f"  Test set : {len(test):,} lignes  (sur {len(df):,} au total)")
print(f"  MAE  = {mae:.3f}  vélos")
print(f"  RMSE = {rmse:.3f}  vélos")

# Bonus : MAE par station, pour repérer les cas faciles vs durs
per_station = (test.assign(err=np.abs(y_true - y_pred))
                    .groupby("station_name")["err"].mean()
                    .sort_values(ascending=False))
print("\nTop 5 stations où la persistance se trompe le plus :")
print(per_station.head().to_string())
print("\nTop 5 stations où la persistance est quasi-parfaite :")
print(per_station.tail().to_string())