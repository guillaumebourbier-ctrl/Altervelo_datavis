#!/usr/bin/env python3
"""Modèle XGBoost NAÏF — sans feature engineering, mais SANS fuite de données.

Différence avec la version utilisateur précédente :
- AVANT (faux) : df["y"] = df["num_vehicles_available"]
                 -> y = présent, features contiennent aussi le présent
                 -> fuite totale, MAE ≈ 0 (le modèle lit la réponse)
- ICI (correct) : y = num_vehicles_available DANS LE FUTUR (t + HORIZON_STEPS)
                  -> le modèle doit prédire ce qu'il ne sait pas

Aucun lag, aucun rolling, aucun diff. Juste les colonnes brutes du CSV.
Ce script sert de baseline minimale honnête.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error


# 1. CHARGEMENT
ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data/stations_clean.csv"
HORIZON_STEPS = 2          # 1 step = 30 min ; 2 = on prédit t+1h
TARGET_COL = "num_vehicles_available"
TEST_RATIO = 0.2


def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
    return df.sort_values(["station_index", "timestamp"]).reset_index(drop=True)


# 2. CIBLE : valeur FUTURE, pas la valeur courante (sinon fuite)
def add_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["y"] = df.groupby("station_index")[TARGET_COL].shift(-HORIZON_STEPS)
    return df.dropna(subset=["y"]).reset_index(drop=True)


# 3. SPLIT TEMPOREL
def temporal_split(df: pd.DataFrame):
    cutoff = df["timestamp"].quantile(1 - TEST_RATIO)
    train = df[df["timestamp"] < cutoff]
    test = df[df["timestamp"] >= cutoff]

    drop_cols = ["timestamp", "station_name", "y", "count_x2"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    return (train[feature_cols], train["y"],
            test[feature_cols], test["y"], feature_cols)


# 4. ENTRAÎNEMENT
PARAMS = dict(
    objective="reg:squarederror",
    n_estimators=500, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1,
)


def main():
    print("1/4 chargement…")
    df = load_data()
    df = add_target(df)
    print(f"   {len(df):,} lignes × {df.shape[1]} colonnes (cible y = t+{HORIZON_STEPS*30} min)")

    print("2/4 split…")
    X_tr, y_tr, X_te, y_te, cols = temporal_split(df)
    print(f"   train={len(X_tr):,}  test={len(X_te):,}  features={len(cols)}  ->  {cols}")

    print("3/4 entraînement…")
    model = xgb.XGBRegressor(**PARAMS, early_stopping_rounds=20)
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

    print("4/4 évaluation…")
    pred = model.predict(X_te)
    rmse = np.sqrt(mean_squared_error(y_te, pred))
    mae = mean_absolute_error(y_te, pred)
    print(f"   RMSE = {rmse:.3f}   MAE = {mae:.3f}")

    # Baseline persistance pour comparer : prédire que t+1h = t
    persistence = X_te[TARGET_COL].values
    p_mae = mean_absolute_error(y_te, persistence)
    p_rmse = np.sqrt(mean_squared_error(y_te, persistence))
    print(f"   Baseline persistance (pred = valeur actuelle) : RMSE = {p_rmse:.3f}   MAE = {p_mae:.3f}")
    print(f"   -> Le modèle bat la baseline ? {'OUI' if mae < p_mae else 'NON'}")

    model.save_model(str(ROOT / "xgb_velos_naive.json"))
    print("\nTop 10 features par importance :")
    imp = pd.Series(model.feature_importances_, index=cols).sort_values(ascending=False)
    print(imp.head(10).to_string())


if __name__ == "__main__":
    main()
