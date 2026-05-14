#!/usr/bin/env python3
"""Variante de train_xgboostplus.py — cible = Δy au lieu de y brut, multi-horizons.

Motivation (cf. Bilan §3 v2) : sur la cible brute `y`, XGBoost dépense 93 %
d'importance sur `current_value` + `lag_1` (la valeur change peu en 1 h).
Conséquence : les features fusion sont vues mais sous-utilisées.

Solution : prédire la *variation* Δy = y(t+h) - y(t).
- L'auto-régression triviale (`current_value`) devient inopérante par construction.
- Les features fusion (transit, batteries) deviennent les seules à porter le signal.
- La baseline persistance change : prédire « pas de changement » <=> Δy = 0.

On compare à 4 horizons (1h, 1h30, 2h, 2h30) pour voir où la prédiction
devient « intéressante » (lift positif vs persistance sur les hubs).

Reporting : top 10 stations difficiles (dérivé dynamiquement par std).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error


# Détection auto du layout : repo (script/ + data/ frères) vs. dossier plat (Colab).
_HERE = Path(__file__).resolve().parent
if (_HERE.parent / "data").exists() or _HERE.name == "script":
    ROOT = _HERE.parent                      # layout repo : script/ → repo racine
else:
    ROOT = _HERE                             # layout plat (Colab /content/) : tout au même niveau

# CSV cherché en priorité dans data/, sinon directement à côté du script (Colab).
_CSV_CANDIDATES = [ROOT / "data/stations_enriched.csv", ROOT / "stations_enriched.csv"]
CSV_PATH = next((p for p in _CSV_CANDIDATES if p.exists()), _CSV_CANDIDATES[0])

# Dossier de sortie : data/ s'il existe, sinon ROOT (Colab).
OUT_DIR = ROOT / "data" if (ROOT / "data").exists() else ROOT
HORIZONS = [2, 3, 4, 5]            # pas de 30 min -> 1h, 1h30, 2h, 2h30
TOP_N_HARD = 10
TEST_RATIO = 0.2

LAGS = [1, 2, 4, 6, 12, 48]
ROLL_WINDOWS = [4, 12, 48]
DIFF_LAGS = [1, 4, 48]
TARGET_COL = "num_vehicles_available"

# Étape 1 : on coupe les features quasi-redondantes avec y(t) pour forcer
# le modèle à apprendre du signal exogène plutôt que recopier l'observation.
#   "full"        : toutes les features lag/roll/diff (v3 historique)
#   "cyclic_only" : ne garde que lag_12/48, roll_*_48, diff_48 (cycles longs)
#   "none"        : aucune feature autoregressive/cyclique — diagnostic pur
#                   pour mesurer ce que les features exogènes (flux, transit,
#                   calendrier) portent comme signal sans aucune persistance possible.
LAG_MODE = "full"   # v3 : cible Δy mais on garde TOUTES les features lag/roll/diff
CYCLIC_DROP_FEATURES = [
    "current_value",
    "lag_1", "lag_2", "lag_4", "lag_6",
    "roll_mean_4", "roll_mean_12",
    "roll_std_4", "roll_std_12",
    "diff_1", "diff_4",
]
# En mode "none", on retire EN PLUS lag_12 (6h, encore proche du présent),
# mais on GARDE les features 24h (lag_48, roll_*_48, diff_48) qui encodent
# le cycle journalier — information cyclique légitime, pas un proxy de
# persistance. C'est cohérent avec l'objectif : tester si les flux portent
# du signal indépendamment des lags courts/moyens.
NONE_DROP_FEATURES = CYCLIC_DROP_FEATURES + ["lag_12"]

PARAMS = dict(
    objective="reg:absoluteerror",  # aligné avec la métrique d'évaluation (MAE)
    n_estimators=500, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1,
    enable_categorical=True,  # Étape 2 : station_index passe en catégoriel natif
)


def _flow_csv_path() -> Path | None:
    """Localise vehicle_flow.csv s'il existe (data/ ou racine plate Colab)."""
    for p in (ROOT / "data/vehicle_flow.csv", ROOT / "vehicle_flow.csv"):
        if p.exists():
            return p
    return None


def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
    flow_path = _flow_csv_path()
    if flow_path is not None:
        flow = pd.read_csv(flow_path, parse_dates=["timestamp"])
        before = df.shape[1]
        df = df.merge(flow, on=["timestamp", "station_index"], how="left")
        added = df.shape[1] - before
        print(f"   features flux mergées depuis {flow_path.name} : +{added} colonnes")

        # Ratios densité-pondérés : normalise les flux par la densité locale du voisinage
        # (somme n_transit dans 0-300m). Idée : à un trafic absolu équivalent, une
        # station isolée subit relativement plus l'arrivée d'1 vélo qu'une station
        # en centre-ville où il y en a déjà 20 dans le coin. Le +1 évite la div/0.
        local_density_cols = [c for c in ["n_transit_0_150m", "n_transit_150_300m"]
                              if c in df.columns]
        if local_density_cols:
            local_density = df[local_density_cols].sum(axis=1)
            arr_60 = [c for c in df.columns if c.startswith("n_arrivees_") and c.endswith("_60min")]
            dep_60 = [c for c in df.columns if c.startswith("n_departs_") and c.endswith("_60min")]
            df["flow_density_arrivees_60min"] = df[arr_60].sum(axis=1) / (1.0 + local_density)
            df["flow_density_departs_60min"] = df[dep_60].sum(axis=1) / (1.0 + local_density)
            print(f"   ratios densité ajoutés : 2 colonnes (sur {len(arr_60)} arrivees + "
                  f"{len(dep_60)} departs / 1+densité_locale)")
    else:
        print("   (vehicle_flow.csv absent — entraînement sans features de flux)")
    return df.sort_values(["station_index", "timestamp"]).reset_index(drop=True)


def time_regime(hour: int) -> int:
    if 1 <= hour < 7:   return 0
    if 7 <= hour < 12:  return 1
    if 12 <= hour < 17: return 2
    if 17 <= hour < 21: return 3
    return 4


def add_features(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Construit les features + cible delta pour un horizon donné."""
    df = df.copy()

    ts = df["timestamp"]
    df["hour"] = ts.dt.hour
    df["dow"] = ts.dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["time_regime"] = df["hour"].apply(time_regime)

    g = df.groupby("station_index")[TARGET_COL]
    for lag in LAGS:
        df[f"lag_{lag}"] = g.shift(lag)
    for w in ROLL_WINDOWS:
        df[f"roll_mean_{w}"] = g.shift(1).rolling(w).mean().reset_index(0, drop=True)
        df[f"roll_std_{w}"] = g.shift(1).rolling(w).std().reset_index(0, drop=True)

    df["current_value"] = df[TARGET_COL]
    for d in DIFF_LAGS:
        df[f"diff_{d}"] = df[TARGET_COL] - g.shift(d)

    # CIBLE = Δy entre t et t+horizon, plutôt que y(t+horizon)
    y_future = g.shift(-horizon)
    df["y_abs"] = y_future                              # gardée pour reconstruction
    df["y"] = y_future - df[TARGET_COL]                 # Δy : la cible réelle apprise

    needed = [f"lag_{l}" for l in LAGS] + [f"diff_{d}" for d in DIFF_LAGS]
    return df.dropna(subset=["y"] + needed).reset_index(drop=True)


def temporal_split(df: pd.DataFrame):
    cutoff = df["timestamp"].quantile(1 - TEST_RATIO)
    df = df.copy()
    # Étape 2 : station_index gardé comme feature catégorielle (pas dans drop_cols).
    df["station_index"] = df["station_index"].astype("category")
    train = df[df["timestamp"] < cutoff]
    test = df[df["timestamp"] >= cutoff]
    drop_cols = ["timestamp", "station_name", "y", "y_abs", TARGET_COL,
                 "count_x2", "lat", "lon"]
    if LAG_MODE == "cyclic_only":
        drop_cols += CYCLIC_DROP_FEATURES
    elif LAG_MODE == "none":
        drop_cols += NONE_DROP_FEATURES
    feature_cols = [c for c in df.columns if c not in drop_cols]
    # current_value reste dans le df (pour reconstruction y_abs) mais hors features.
    current_te = test[TARGET_COL].astype(float).values
    return (train[feature_cols], train["y"],
            test[feature_cols], test["y"], test["y_abs"],
            current_te, feature_cols)


def hardest_stations(df: pd.DataFrame, n: int) -> list[str]:
    g = df.groupby("station_name")[TARGET_COL].std().sort_values(ascending=False)
    return g.head(n).index.tolist()


def report(X_test, current, y_delta_true, delta_pred, y_abs_true, station_names, hard_names):
    """Reporting sur les top N hardest stations.
    On évalue en valeur absolue reconstruite : y_pred = current_value + Δy_pred.
    Persistance reconstruite équivalente : y_pers = current_value (Δy=0).
    """
    y_pred_abs = current + delta_pred

    df_eval = pd.DataFrame({
        "station_index": X_test["station_index"].values,
        "y_true": y_abs_true.values,
        "y_pred": y_pred_abs,
        "y_pers": current,
    })
    df_eval["station_name"] = df_eval["station_index"].map(station_names)
    df_eval["err_model"] = (df_eval["y_true"] - df_eval["y_pred"]).abs()
    df_eval["err_pers"]  = (df_eval["y_true"] - df_eval["y_pers"]).abs()

    g = df_eval.groupby("station_name").agg(
        n=("y_true", "size"), std=("y_true", "std"),
        mae_model=("err_model", "mean"), mae_pers=("err_pers", "mean"),
    )
    g["lift_%"] = (g["mae_pers"] - g["mae_model"]) / g["mae_pers"].replace(0, np.nan) * 100
    print(g.reindex(hard_names).round(3).to_string())


def run_one_horizon(df_full: pd.DataFrame, horizon: int, hard_names: list[str], station_names: dict):
    label = f"{horizon * 30} min"
    print(f"\n========== HORIZON = {label} (h={horizon} pas) ==========")
    df = add_features(df_full, horizon)
    X_tr, y_tr, X_te, y_te, y_abs_te, current, cols = temporal_split(df)
    print(f"   train={len(X_tr):,}  test={len(X_te):,}  features={len(cols)}  "
          f"(LAG_MODE={LAG_MODE})")

    model = xgb.XGBRegressor(**PARAMS, early_stopping_rounds=20)
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    suffix = {"full": "", "cyclic_only": "", "none": "_nolag"}[LAG_MODE]
    out_path = OUT_DIR / f"xgb_velos_v4{suffix}_h{horizon * 30}min.json"
    model.save_model(str(out_path))
    print(f"   modèle sauvegardé → {out_path}")

    delta_pred = model.predict(X_te)

    # Métriques GLOBALES (sur la valeur absolue reconstruite, comparable à v1/v2)
    y_pred_abs = current + delta_pred
    mae_model = mean_absolute_error(y_abs_te, y_pred_abs)
    mae_pers = mean_absolute_error(y_abs_te, current)
    lift = (mae_pers - mae_model) / mae_pers * 100 if mae_pers > 0 else float("nan")
    print(f"   MAE (global) modèle = {mae_model:.3f}   persistance = {mae_pers:.3f}   "
          f"lift = {lift:+.1f}%   {'OUI' if mae_model < mae_pers else 'NON'}")

    # Anticipation : sur les moments où ça bouge vraiment, est-ce que delta_pred
    # corrèle avec delta_true ? Si ≈0 → le modèle se contente de persister.
    # On rapporte 3 seuils : |Δ|≥1 (tout évènement réel, vélos = entiers),
    # |Δ|≥2 (mouvement substantiel), |Δ|≥3 (gros pics, surtout hubs).
    y_delta = y_te.values
    anticip_by_thr = {}
    for thr in (1, 2, 3):
        mask = np.abs(y_delta) >= thr
        if mask.sum() > 1 and np.std(delta_pred[mask]) > 0:
            corr = float(np.corrcoef(delta_pred[mask], y_delta[mask])[0, 1])
        else:
            corr = float("nan")
        anticip_by_thr[thr] = (corr, int(mask.sum()))
    print("   Anticipation score (corr Δ_pred vs Δ_true) :")
    for thr, (corr, n) in anticip_by_thr.items():
        print(f"      |Δ|≥{thr}  n={n:>5}   corr={corr:+.3f}")
    anticip = anticip_by_thr[1][0]  # référence pour la synthèse multi-horizons

    print(f"\n   === Top {TOP_N_HARD} stations DIFFICILES (par std) ===")
    report(X_te, current, y_te, delta_pred, y_abs_te, station_names, hard_names)

    # Top features
    imp = pd.Series(model.feature_importances_, index=cols).sort_values(ascending=False)
    print(f"\n   Top 10 features :")
    print(imp.head(10).to_string())

    return {
        "horizon_min": horizon * 30,
        "mae_model": mae_model,
        "mae_pers": mae_pers,
        "lift_%": lift,
        "anticip": anticip,
        "top1_feat": imp.index[0],
        "top1_imp": imp.iloc[0],
    }


def main():
    print("Chargement…")
    df_full = load_data()
    station_names = dict(zip(df_full["station_index"], df_full["station_name"]))

    hard_names = hardest_stations(df_full, TOP_N_HARD)
    print(f"Top {TOP_N_HARD} stations difficiles (par std globale) :")
    for i, name in enumerate(hard_names, 1):
        print(f"   {i:>2}. {name}")

    rows = []
    for h in HORIZONS:
        rows.append(run_one_horizon(df_full, h, hard_names, station_names))

    print("\n========== SYNTHÈSE MULTI-HORIZONS ==========")
    summary = pd.DataFrame(rows)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
