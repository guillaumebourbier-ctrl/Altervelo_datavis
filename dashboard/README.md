# Dashboard AlterVélo

Dashboard de mise en service des modèles XGBoost entraînés dans le projet parent.
Trois pages : **état actuel**, **prévision** à 60-150 min, **monitoring** de la qualité du modèle dans le temps.

> Ce dossier est volontairement autonome (`venv` + `velos.db` + `requirements.txt` dédiés) pour
> séparer le travail produit (servir le modèle) du travail recherche (entraîner le modèle)
> qui vit à la racine du projet.

---

## Déploiement

Ce dossier est **autonome** : il ne contient ni `.venv` (1.3 Go local), ni `velos.db`
(régénérée), ni `pipeline.log` — tout est dans `.gitignore`. Le dépôt poussé pèse
< 1 Mo. Suivre les étapes ci-dessous pour redéployer depuis zéro.

### 1. Prérequis

- Python ≥ 3.11
- Connexion sortante vers l'API GBFS publique du réseau de vélos
- Les 4 modèles XGBoost entraînés : `xgb_velos_v3_h{60,90,120,150}min.json`
  → à placer dans le dossier **parent** de `dashboard/` (ce dépôt sert les modèles, il ne les entraîne pas)

### 2. Cloner & installer

```bash
git clone <url-du-repo> alter-velos
cd alter-velos/dashboard

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 3. Bootstrap de la base SQLite

Deux modes selon ce qu'on a sous la main.

**a) Démarrage à froid (recommandé sur GitHub)** — crée les tables vides, puis
le pipeline d'ingestion remplit la DB au fil des ticks GBFS :

```bash
.venv/bin/python ingest/schema.py     # idempotent : crée les tables si absentes
bash ingest/pipeline.sh               # premier tick → seed initial de velos.db
```

**b) Reconstruction complète à partir des CSV historiques** — uniquement si
les CSV du projet parent (`../data/stations_enriched.csv`) sont disponibles :

```bash
.venv/bin/python db_init.py           # ETL + backtest 4 horizons (destructif)
```

### 4. Lancement

```bash
# UI Streamlit (port 8501 par défaut)
.venv/bin/streamlit run app/dashboard.py

# Pipeline en boucle (tick toutes les 30 min)
nohup bash -c 'while true; do bash ingest/pipeline.sh; sleep 1800; done' \
    >> pipeline.log 2>&1 &
```

### 5. Production (systemd / tmux)

Exemple unité systemd minimaliste (`/etc/systemd/system/altervelo-ingest.service`) :

```ini
[Service]
Type=simple
WorkingDirectory=/opt/alter-velos/dashboard
ExecStart=/bin/bash -c 'while true; do bash ingest/pipeline.sh; sleep 1800; done'
Restart=on-failure
StandardOutput=append:/opt/alter-velos/dashboard/pipeline.log
StandardError=append:/opt/alter-velos/dashboard/pipeline.log

[Install]
WantedBy=multi-user.target
```

Pour l'UI on enchaînera derrière un reverse-proxy (nginx) pointant sur
`127.0.0.1:8501`, le binaire Streamlit étant lancé via une seconde unité systemd.

### Notes

- `dashboard/pipeline.sh` (à la racine du dossier) est **obsolète** — il `cd ..`
  dans la chaîne CSV du projet parent. Toujours utiliser `ingest/pipeline.sh`.
- Les fichiers `velos.db`, `pipeline.log` et `.venv/` sont volontairement
  exclus du repo : tout est reproductible depuis `requirements.txt` +
  `ingest/schema.py` + l'API GBFS.

## Architecture

```
                  ┌──────── racine projet ────────┐
                  │  collect.py    cleaning_*.py  │
                  │  merge_csv.py  train_*.py     │
                  │  data/*.csv    *.json modèles │
                  └────────────┬──────────────────┘
                               │
                               ▼  (CSV + modèles JSON)
                  ┌──────── dashboard/ ───────────┐
                  │   db_init.py   (one-shot ETL) │
                  │   predict.py   (cron tick)    │
                  │   evaluate.py  (backfill)     │
                  │            │                  │
                  │            ▼                  │
                  │   velos.db (SQLite)           │
                  │   ├ stations                  │
                  │   ├ observations  (live + h.) │
                  │   ├ predictions   (live + bt) │
                  │   └ pipeline_runs             │
                  │            │                  │
                  │            ▼                  │
                  │   app/dashboard.py (Streamlit)│
                  │   ├ page_t0       (état)      │
                  │   ├ page_pred     (prévision) │
                  │   └ page_monit    (qualité)   │
                  └───────────────────────────────┘
```

## Schéma SQL

```sql
stations(station_index PK, station_name, lat, lon, capacity)

observations(timestamp, station_index, ..., source ['historical' | 'live'])
  PK (timestamp, station_index)

predictions(ts_pred, ts_target, horizon_min, station_index,
            y_pred, y_current, y_obs?, err_model?, err_pers?,
            source ['backtest' | 'live'])
  PK (ts_pred, horizon_min, station_index)

pipeline_runs(ts_run PK, duration_ms, n_*, status, error_msg)
```

Pattern « delayed feedback » : `y_obs/err_*` sont NULL à l'INSERT, remplis par
`evaluate.py` quand l'observation cible (`ts_target`) arrive en base.

## Pages

| Page | Source SQL | Vue principale |
|---|---|---|
| **T-0** | dernier `MAX(timestamp)` de `observations` | carte coloriée par % de remplissage |
| **Prévision** | dernière `MAX(ts_pred)` live de `predictions` (par horizon) | carte coloriée par alerte (`rupture`/`tension`/`saturation`/`normal`) |
| **Monitoring** | toute la table `predictions` (backtest + live) groupée | MAE rolling + carte stations difficiles + tableau winners/loosers |

Le seuil de classification d'alerte (`classify_alert` dans `views/page_pred.py`)
est paramétré par `K_VOLATILITY = 1.5` (un mouvement est « anormal » au-delà
de 1.5 × σ historique de la station).

## Limites connues

- **Auto-régression dominante à 60 min** : MAE modèle 0.092 vs persistance 0.086 (lift -6 %).
  C'est le bayésien optimal en MAE sur série quasi-stationnaire (cf. Bilan §3 v3).
  Le modèle bat la persistance significativement uniquement sur les hubs (Port, Casabona).
- **Le clip `np.clip(y_pred, 0, None)`** ajoute ~5 % de MAE relatif vs Colab brut.
  Trade-off conscient : pas de prédiction négative dans le dashboard.
- **`time_regime` doit rester aligné** entre `dashboard/features.py` et
  `train_xgboostplus_delta.py`. Toute divergence = train/serve skew silencieux.
- **Cron 30 min** doit tourner pendant la durée d'observation pour que la page
  Monitoring se peuple en live au-delà du backtest historique.

## Captures

À ajouter une fois le dashboard lancé :
- `docs/screen_t0.png` — carte des stations à T-0
- `docs/screen_pred.png` — projection T+60min avec alertes
- `docs/screen_monit.png` — courbe MAE rolling + tableau debug
