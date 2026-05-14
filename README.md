# AlterVélo — Prévision de disponibilité des vélos

> Collecte GBFS, prétraitement, entraînement XGBoost multi-horizons et dashboard Streamlit de prévision/monitoring pour AlterVélo (La Réunion).

## Vue d'ensemble

Pipeline complet de bout en bout sur les données du service de vélos en libre-service AlterVélo (La Réunion) :

1. **Collecte** — appels périodiques aux API GBFS v3.0 (positions, stations, types, tarifs)
2. **Prétraitement** — nettoyage, gestion des gaps, fusion et enrichissement des CSV
3. **Entraînement** — modèles XGBoost entraînés à prédire la *variation* de disponibilité (Δy) à 4 horizons : 60, 90, 120 et 150 min
4. **Dashboard** — application Streamlit autonome (état T-0, prévisions avec alertes, monitoring MAE rolling)

---

## Sources de données

| API | Description | CSV généré |
|-----|-------------|------------|
| [vehicle_status](https://api.gbfs.v3.0.ecovelo.mobi/altervelo/vehicle_status.json) | Position, batterie, autonomie et état de chaque vélo | `data/vehicle_status.csv` |
| [station_status](https://api.gbfs.v3.0.ecovelo.mobi/altervelo/station_status.json) | Nombre de docks et vélos disponibles par station | `data/station_status.csv` |
| [system_pricing_plans](https://api.gbfs.v3.0.ecovelo.mobi/altervelo/system_pricing_plans.json) | Plans tarifaires (prix à la minute, paliers) | `data/pricing_plans.csv` |
| [vehicle_types](https://api.gbfs.v3.0.ecovelo.mobi/vehicle_types.json) | Types de vélos (forme, propulsion, autonomie max) | `data/vehicle_types.csv` |

L'API GBFS est temps réel uniquement (pas d'historique) — la collecte périodique via `collect.py` est donc indispensable pour constituer un jeu de données exploitable.

---

## Collecte automatique

`collect.py` appelle les 4 API et **ajoute** les résultats dans les CSV correspondants avec un horodatage UTC.

```bash
pip install requests
python collect.py
```

Un timer systemd utilisateur exécute la collecte toutes les 2 heures (`Persistent=true` relance au réveil si le PC était éteint). Les fichiers de configuration se trouvent dans `~/.config/systemd/user/` :
- `collect-velos.service`
- `collect-velos.timer`

```bash
systemctl --user status collect-velos.timer
systemctl --user start collect-velos.service
journalctl --user -u collect-velos.service
```

---

## Prétraitement (`script/`)

| Script | Rôle |
|--------|------|
| `cleaning_data.py` | Nettoyage général, typage, déduplication |
| `clean_vehicle_status.py` | Nettoyage spécifique véhicules, suppression fantômes |
| `fill_gaps.py` | Interpolation des gaps (ffill nuit, interpolation linéaire jour) |
| `merge_csv.py` | Fusion station + véhicules, calcul bandes haversine transit |
| `build_vehicle_flow.py` | Flux d'entrées/sorties par station |

Résultat : `data/stations_enriched.csv` — jeu de données consolidé utilisé pour l'entraînement.

---

## Entraînement XGBoost (`script/`)

| Script | Rôle |
|--------|------|
| `train_xgboostplus_delta_v3.py` | Entraînement principal (cible Δy, 4 horizons) |
| `train_naive.py` | Baseline naïf |
| `baseline_persistance.py` | Baseline persistance (référence) |

**Choix de conception — cible Δy :** sur la cible brute `y`, XGBoost dépense 93 % d'importance sur `current_value` + `lag_1`. En prédisant la *variation* Δy = y(t+h) − y(t), l'auto-régression triviale devient inopérante et les features fusion (transit, batteries) portent le signal.

Les modèles entraînés sont exportés dans `model/` :

```
model/
├── xgb_velos_v3_h60min.json
├── xgb_velos_v3_h90min.json
├── xgb_velos_v3_h120min.json
└── xgb_velos_v3_h150min.json
```

---

## Dashboard Streamlit (`dashboard/`)

Application autonome (venv + SQLite propres) qui sert les modèles entraînés en production. Voir [`dashboard/README.md`](dashboard/README.md) pour le déploiement complet.

**Trois pages :**
- **T-0** — carte des stations colorée par taux de remplissage actuel
- **Prévision** — projections à 60/90/120/150 min avec alertes (`rupture` / `tension` / `saturation` / `normal`)
- **Monitoring** — MAE rolling, stations difficiles, comparaison modèle vs persistance

Pipeline live : tick toutes les 30 min → collecte GBFS → nettoyage → SQLite → prédiction → évaluation différée.

```bash
cd dashboard
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python ingest/schema.py
bash ingest/pipeline.sh
.venv/bin/streamlit run app/dashboard.py
```

---

## Structure du projet

```
.
├── collect.py                     # Collecte GBFS → CSV
├── README.md
├── Altervelo.pdf                  # Document de référence
├── data/
│   ├── vehicle_status.csv
│   ├── station_status.csv
│   ├── pricing_plans.csv
│   ├── vehicle_types.csv
│   └── stations_enriched.csv      # Jeu consolidé (issu de script/)
├── script/                        # Prétraitement + entraînement
│   ├── cleaning_data.py
│   ├── clean_vehicle_status.py
│   ├── fill_gaps.py
│   ├── merge_csv.py
│   ├── build_vehicle_flow.py
│   ├── train_xgboostplus_delta_v3.py
│   ├── train_naive.py
│   └── baseline_persistance.py
├── model/                         # Modèles XGBoost exportés
│   └── xgb_velos_v3_h{60,90,120,150}min.json
├── notebook/                      # Exploration interactive
├── latex/                         # Rapport
├── ressources/
└── dashboard/                     # Application Streamlit autonome
    ├── ingest/                    # Pipeline live (GBFS → SQLite)
    ├── app/                       # UI Streamlit (3 pages)
    ├── predict.py
    ├── evaluate.py
    ├── db.py
    └── README.md
```
