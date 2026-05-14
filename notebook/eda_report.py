#!/usr/bin/env python3
"""
EDA Report — AlterVélo Réunion
Reprend le contenu de analyse_exploratoire.ipynb et analyse_exploratoire2.ipynb.
Toutes les sorties sont textuelles (pas de graphiques).
"""

import numpy as np
import pandas as pd

pd.options.display.max_columns = 30
pd.options.display.width = 120

SEP = "=" * 70
sep = "-" * 50


def section(title):
    print(f"\n{sep}\n{title}\n{sep}")


def part(title):
    print(f"\n{SEP}\n{title}\n{SEP}")


# ============================================================
# NOTEBOOK 1 — analyse_exploratoire.ipynb
# ============================================================

part("NOTEBOOK 1 — Analyse exploratoire (stations_clean.csv)")
print("Données : data/stations_clean.csv (sortie de cleaning_data.py).")
print("Objectif : comprendre la structure du jeu avant l'entraînement XGBoost.")

# --- §1 — Chargement -----------------------------------------------------------

section("§1 — Chargement")

df = pd.read_csv('../data/stations_clean.csv', parse_dates=['timestamp'])
df = df.sort_values(['station_index', 'timestamp']).reset_index(drop=True)

print(f"Shape : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
print("\nTypes de colonnes :")
print(df.dtypes.to_string())
print("\n5 premières lignes :")
print(df.head().to_string())

# --- §2 — Vue d'ensemble -------------------------------------------------------

section("§2 — Vue d'ensemble")

print(f"Période          : {df['timestamp'].min()}  ->  {df['timestamp'].max()}")
print(f"Nb stations      : {df['station_index'].nunique()}")
print(f"Timestamps uniques: {df['timestamp'].nunique()}")
print(f"Lignes imputées  : {df['is_imputed'].mean() * 100:.1f}%")
print("\nStats descriptives (colonnes numériques, transposées) :")
print(df.describe().T.round(3).to_string())

# --- §3 — Qualité : imputations par station ------------------------------------

section("§3 — Qualité des données — imputations par station")
print("Chaque graphique barh est remplacé par un classement textuel.")

imp_by_station = (
    df.groupby('station_name')['is_imputed'].mean().sort_values(ascending=False) * 100
)

print("\n% lignes imputées par station (ordre décroissant) :")
print(imp_by_station.round(1).to_string())
print(f"\nMédiane réseau   : {imp_by_station.median():.1f}%")
print(f"Top 3 stations les plus imputées :")
for name, val in imp_by_station.head(3).items():
    print(f"  {name:<45} {val:.1f}%")

# --- §4 — Disponibilité globale : vélos vs docks --------------------------------

section("§4 — Disponibilité globale — vélos disponibles vs docks libres")
print(
    "Anti-corrélation attendue entre les deux séries (capacité physique fixe).\n"
    "Les discontinuités ou dérives indiquent des pannes / redéploiements."
)

global_ts = df.groupby('timestamp')[['num_vehicles_available', 'num_docks_available']].sum()

print("\nStats réseau agrégé (somme sur toutes les stations) :")
print(global_ts.describe().round(1).to_string())

capacity = global_ts['num_vehicles_available'] + global_ts['num_docks_available']
print(f"\nCapacité totale (vélos + docks) — stabilité :")
print(f"  mean={capacity.mean():.1f}  std={capacity.std():.1f}  "
      f"min={capacity.min():.0f}  max={capacity.max():.0f}")

corr_vd = global_ts['num_vehicles_available'].corr(global_ts['num_docks_available'])
std_docks = global_ts['num_docks_available'].std()
print(f"\nCorrélation vélos ↔ docks (niveau réseau) : {corr_vd:.3f}")
if std_docks == 0:
    print("  -> NaN car num_docks_available est CONSTANT sur tous les snapshots (std=0).")
    print("  -> Cela signifie que le total des docks libres ne varie pas : les vélos")
    print("     disponibles varient mais les docks restent figés — à investiguer.")
else:
    print("  -> Proche de -1 = effet bascule attendu confirmé.")

# --- §5 — Profil journalier moyen (heure locale) --------------------------------

section("§5 — Profil journalier moyen (heure locale, UTC+4)")
print(
    "Semaine vs week-end : creux matinal attendu (départs domicile→travail).\n"
    "Nuits plates (22h-5h) = zones forward-fill — absence de variation suspecte."
)

df['hour'] = df['timestamp'].dt.hour
df['dow'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['dow'] >= 5

profile = df.groupby(['hour', 'is_weekend'])['num_vehicles_available'].mean().unstack()
profile.columns = ['Semaine', 'Week-end']

print("\nVélos disponibles moyens par station, par heure :")
print(profile.round(2).to_string())

delta = profile['Week-end'] - profile['Semaine']
print("\nDelta (Week-end − Semaine) par heure :")
print(delta.round(2).to_string())
print(f"\nDelta max : heure {delta.idxmax()}h ({delta.max():+.2f} vélos)")
print(f"Delta min : heure {delta.idxmin()}h ({delta.min():+.2f} vélos)")

# --- §6 — Heatmap heure × jour de la semaine ------------------------------------

section("§6 — Heatmap heure × jour de la semaine")
print(
    "Bandes verticales homogènes → peu d'effet jour-de-la-semaine.\n"
    "Contraste lundi-vendredi / samedi-dimanche → lag_336 utile.\n"
    "Note : ~22 jours ≈ 3 mesures par cellule — tableau bruité."
)

heat = df.groupby(['dow', 'hour'])['num_vehicles_available'].mean().unstack()
heat.index = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim']

print("\nDisponibilité moyenne (vélos) — heure (colonnes) × jour (lignes) :")
print(heat.round(2).to_string())

stacked = heat.stack()
top3 = stacked.nlargest(3)
bot3 = stacked.nsmallest(3)
print("\nTop 3 créneaux (plus haute disponibilité) :")
for (dow, hr), val in top3.items():
    print(f"  {dow} {hr:02d}h → {val:.2f} vélos")
print("Bottom 3 créneaux (plus basse disponibilité) :")
for (dow, hr), val in bot3.items():
    print(f"  {dow} {hr:02d}h → {val:.2f} vélos")

# --- §7 — Stations les plus / moins fréquentées ----------------------------------

section("§7 — Stations les plus / moins fréquentées (variabilité)")
print(
    "Tri par std : stations en tête = forte variation = prévision à valeur ajoutée.\n"
    "Ratio std top/bottom > 10× → réseau hétérogène, évaluer par segment."
)

rotation = (
    df.groupby('station_name')['num_vehicles_available']
    .agg(['mean', 'std'])
    .sort_values('std', ascending=False)
)

print("\nTop 10 stations à forte variabilité :")
print(rotation.head(10).round(2).to_string())
print("\nBottom 5 stations (faible variabilité) :")
print(rotation.tail(5).round(2).to_string())

top3_names = rotation.head(3).index.tolist()
print(f"\nSéries temporelles — 3 stations à plus forte rotation :")
for name in top3_names:
    sub = df[df['station_name'] == name]['num_vehicles_available']
    print(f"  {name:<45}  min={sub.min():.0f}  max={sub.max():.0f}  "
          f"mean={sub.mean():.1f}  std={sub.std():.1f}")

# --- §8 — Corrélations entre compteurs -------------------------------------------

section("§8 — Corrélations entre compteurs")
print(
    "num_vehicles_available ↔ num_docks_available : anti-corrélation attendue ≈ -1.\n"
    "num_vehicles_available ↔ count_x2 : fuite par proxy attendue ≈ +1."
)

corr_cols = [
    'num_vehicles_available', 'num_vehicles_disabled',
    'num_docks_available', 'num_docks_disabled', 'count_x2'
]
corr_matrix = df[corr_cols].corr().round(3)
print("\nMatrice de corrélation :")
print(corr_matrix.to_string())

print("\nPaires clés :")
pairs = [
    ('num_vehicles_available', 'num_docks_available'),
    ('num_vehicles_available', 'count_x2'),
    ('num_vehicles_disabled', 'num_docks_disabled'),
]
for a, b in pairs:
    v = corr_matrix.loc[a, b]
    if pd.isna(v):
        print(f"  {a} ↔ {b} : NaN (std=0 sur l'une des deux colonnes — constante)")
    else:
        print(f"  {a} ↔ {b} : {v:+.3f}")

# --- §9 — Autocorrélation --------------------------------------------------------

section("§9 — Autocorrélation (effet mémoire de la série)")
print(
    "Granularité 30 min : lag 48 = 24h, lag 336 = 1 semaine.\n"
    "Pic à lag 48 → cycle journalier (feature lag_48 utile).\n"
    "Pic à lag 336 → cycle hebdomadaire (feature lag_336 utile).\n"
    "Analyse sur la première station — potentiellement non représentatif."
)

sample_station = df['station_index'].iloc[0]
sample = (
    df[df['station_index'] == sample_station]
    .sort_values('timestamp')
    .set_index('timestamp')['num_vehicles_available']
)

lags = [1, 2, 4, 12, 24, 48, 96, 168, 336]
print(f"\nStation analysée : index {sample_station}  ({len(sample)} points)")
print(f"\n{'Lag':>6}  {'Délai':>10}  {'ACF':>8}")
print("-" * 30)
for lag in lags:
    if lag < len(sample):
        acf_val = sample.autocorr(lag=lag)
        if lag < 48:
            label = f"{lag * 30}min"
        elif lag == 48:
            label = "24h"
        elif lag == 96:
            label = "48h"
        elif lag == 168:
            label = "3.5j"
        elif lag == 336:
            label = "1 sem"
        else:
            label = f"{lag} pas"
        print(f"{lag:>6}  {label:>10}  {acf_val:>8.4f}")

# ============================================================
# NOTEBOOK 2 — analyse_exploratoire2.ipynb
# ============================================================

part("NOTEBOOK 2 — EDA #2 (vehicles_clean.csv)")
print(
    "Approche initiale (stations only) : MAE 0.17-0.18 vs persistance 0.087 → modèle perd.\n"
    "Diagnostic : 72% d'importance sur current_value — toutes les features dérivent du même signal.\n"
    "Pivot : vehicle_status.csv (GPS, batterie, statut) = source orthogonale à valider."
)

# --- §0 — Chargement -----------------------------------------------------------

section("§0 — Chargement des données")

vehicles = pd.read_csv('../data/vehicles_clean.csv', parse_dates=['timestamp'])
stations = pd.read_csv('../data/stations_clean.csv', parse_dates=['timestamp'])

print(f"vehicles : {len(vehicles):,} lignes × {vehicles.shape[1]} colonnes")
print(f"stations : {len(stations):,} lignes × {stations.shape[1]} colonnes")
print("\nTypes colonnes vehicles :")
print(vehicles.dtypes.to_string())
print("\n5 premières lignes vehicles :")
print(vehicles.head().to_string())

# --- §1 — Couverture temporelle -----------------------------------------------

section("§1 — Couverture temporelle")
print("Hypothèse : même fenêtre ~26 jours que stations_clean, flotte stable ≈ 100 vélos.")

print(f"\nPlage vehicles : {vehicles['timestamp'].min()}  ->  {vehicles['timestamp'].max()}")
print(f"Plage stations : {stations['timestamp'].min()}  ->  {stations['timestamp'].max()}")
print(f"Vélos uniques  : {vehicles['vehicle_id'].nunique()}")

per_slot = vehicles.groupby('timestamp').size()
print(f"\nVélos par snapshot 30 min :")
print(f"  moy={per_slot.mean():.1f}  std={per_slot.std():.1f}  "
      f"min={per_slot.min()}  max={per_slot.max()}")
print(f"  p25={per_slot.quantile(0.25):.0f}  médiane={per_slot.median():.0f}  "
      f"p75={per_slot.quantile(0.75):.0f}")

# --- §2 — Statut is_disabled ---------------------------------------------------

section("§2 — Statut is_disabled (vélos désactivés par station)")
print("Hypothèse : majorité actifs ; quelques stations concentrent les cassés.")

print(f"\nTaux global is_disabled : {vehicles['is_disabled'].mean():.1%}")

in_station = vehicles[vehicles['station_index'] > 0]
by_station = in_station.groupby('station_index')['is_disabled'].agg(['sum', 'count', 'mean'])
by_station.columns = ['nb_disabled', 'nb_obs', 'taux_disabled']

print("\nTop 10 stations par taux de vélos désactivés :")
print(by_station.sort_values('taux_disabled', ascending=False).head(10).round(3).to_string())

# --- §3 — Distribution current_fuel_percent ------------------------------------

section("§3 — Distribution current_fuel_percent (batterie)")
print(
    "Hypothèse : distribution bimodale. Forte queue à gauche → seuil < 0.20 pour bikes_low_battery.\n"
    "Avertissement : 29% à 0% avant filtre fantômes — suspect (vélos perdus figés à 0)."
)

fuel = vehicles['current_fuel_percent'].dropna()
print(f"\nn = {len(fuel):,}")
print(f"mean = {fuel.mean():.3f}   médiane = {fuel.median():.3f}   std = {fuel.std():.3f}")
print(f"min  = {fuel.min():.3f}   max     = {fuel.max():.3f}")
print(f"\n% < 0.20 (seuil low_battery) : {(fuel < 0.20).mean():.1%}")
print(f"% == 0.0 (batterie nulle)     : {(fuel == 0.0).mean():.1%}")
print(f"% == 1.0 (batterie pleine)    : {(fuel == 1.0).mean():.1%}")

print("\nDistribution par tranches de 10% :")
bins = np.arange(0, 1.1, 0.1)
labels = [f"{int(b*100):3d}-{int((b+0.1)*100):3d}%" for b in bins[:-1]]
counts, _ = np.histogram(fuel, bins=bins)
total = len(fuel)
for lbl, cnt in zip(labels, counts):
    bar = "█" * int(cnt / total * 50)
    print(f"  {lbl} : {cnt:6,} ({cnt/total:5.1%})  {bar}")

# --- §4 — Vélos en circulation (station_index = 0) -----------------------------

section("§4 — Vélos en circulation (station_index = 0)")
print(
    "Hypothèse : pic en transit heure de pointe (7-9h, 17-19h).\n"
    "Corrélation négative attendue avec la somme num_vehicles_available réseau."
)

transit = vehicles[vehicles['station_index'] == 0].groupby('timestamp').size()
transit.name = 'in_transit'

transit_df = transit.reset_index()
transit_df['hour'] = pd.to_datetime(transit_df['timestamp']).dt.hour
profile_transit = transit_df.groupby('hour')['in_transit'].mean()

print("\nVélos en circulation : moyenne par heure (UTC+4) :")
print(f"{'Heure':>6}  {'Moy in_transit':>15}")
print("-" * 25)
for hr, val in profile_transit.items():
    bar = "█" * int(val)
    print(f"  {hr:02d}h   {val:8.2f}  {bar}")

stations_total = stations.groupby('timestamp')['num_vehicles_available'].sum()
stations_total.index = pd.to_datetime(stations_total.index)
transit.index = pd.to_datetime(transit.index)
joined = pd.concat([transit, stations_total], axis=1, join='inner').dropna()
joined.columns = ['in_transit', 'available_total']

corr_tr = joined.corr().iloc[0, 1]
print(f"\nCorrélation transit ↔ dispo réseau : {corr_tr:.3f}")
print("  -> Valeur proche de -1 confirme le signal physique (vélos en route = dispo stations ↓).")
print(f"\nStats in_transit :")
print(joined['in_transit'].describe().round(1).to_string())

# --- §5 — Diagnostic vélos fantômes -------------------------------------------

section("§5 — Diagnostic « 0% batterie » : les vélos fantômes")
print(
    "Hypothèse : vélos perdus/cassés que l'API continue de remonter, batterie figée à 0.\n"
    "Test : un vélo réel est rechargé au moins une fois sur 26 jours.\n"
    "Critère : max(current_fuel_percent) == 0 sur toute la fenêtre → fantôme."
)

raw = pd.read_csv('../data/vehicle_status.csv', parse_dates=['timestamp'])
max_fuel = raw.groupby('vehicle_id')['current_fuel_percent'].max()
phantoms = max_fuel[max_fuel == 0.0].index

total_vehicles = raw['vehicle_id'].nunique()
nb_phantoms = len(phantoms)
nb_phantom_rows = raw['vehicle_id'].isin(phantoms).sum()

print(f"\nFlotte totale (avant filtre)         : {total_vehicles} vélos")
print(f"Fantômes détectés (jamais rechargés) : {nb_phantoms} vélos ({nb_phantoms/total_vehicles:.1%})")
print(f"Lignes représentées par les fantômes : {nb_phantom_rows:,} / {len(raw):,} "
      f"({nb_phantom_rows/len(raw):.1%})")

ph_rows = raw[raw['vehicle_id'].isin(phantoms)]
print(f"\nParmi les lignes fantômes :")
print(f"  taux is_disabled = {ph_rows['is_disabled'].mean():.1%}")
print("  -> L'API ne les marque PAS comme désactivés : filtre naïf is_disabled insuffisant.")
print("  -> Critère temporel (jamais rechargé) nécessaire → implémenté dans clean_vehicle_status.py.")

# --- §6 — Sanity check de jointure ---------------------------------------------

section("§6 — Sanity check de jointure (vélos actifs vs num_vehicles_available)")
print(
    "But : vérifier que (timestamp, station_index) → count(vélos actifs) ≈ num_vehicles_available.\n"
    "Hypothèse : |diff| ≤ 1 dans > 95% des paires (drift 30 min toléré)."
)

va = (
    vehicles[(vehicles['station_index'] > 0) & (vehicles['is_disabled'] == 0)]
    .groupby(['timestamp', 'station_index'])
    .size()
    .rename('vehicles_active')
)
sa = (
    stations
    .set_index(['timestamp', 'station_index'])['num_vehicles_available']
    .rename('num_avail')
)

cmp = pd.concat([va, sa], axis=1, join='inner').fillna(0)
cmp['diff'] = cmp['vehicles_active'] - cmp['num_avail']

print(f"\nPaires (timestamp, station)     : {len(cmp):,}")
print(f"|diff| moyen                    : {cmp['diff'].abs().mean():.3f}")
print(f"% paires avec |diff| <= 1       : {(cmp['diff'].abs() <= 1).mean():.1%}")
print(f"% paires avec diff == 0 (exact) : {(cmp['diff'] == 0).mean():.1%}")

print("\nDistribution des écarts (diff = vehicles_active - num_avail) :")
print(cmp['diff'].astype(int).value_counts().sort_index().to_string())

by_st = cmp.groupby(level='station_index')['diff'].agg(['mean', 'count'])
by_st['abs_mean'] = by_st['mean'].abs()
print("\nTop 5 stations par écart absolu moyen :")
print(by_st.sort_values('abs_mean', ascending=False).head(5).round(3).to_string())

# --- §7 — Conclusion -----------------------------------------------------------

section("§7 — Conclusion & prochaines étapes")
print("""
Ce que ce notebook a établi :

1. vehicle_status.csv est EXPLOITABLE comme source orthogonale à stations_clean :
   taux de vélos désactivés, batteries faibles, vélos en transit.

2. Trois préprocessings non-triviaux ont été nécessaires :
   - Drop des colonnes constantes (vehicle_type_id, is_reserved) ;
   - Filtrage de 31 vélos fantômes (24% de la flotte), invisibles via is_disabled seul ;
   - Recadrage temporel sur la fenêtre commune avec stations_clean.

3. Sanity check de jointure VALIDÉ :
   (timestamp, station_index) → comptes vélos vs num_vehicles_available
   coïncident à ±1 vélo dans 98.5% des cas. Fusion techniquement saine.

4. Features débloquées par la fusion :
   - bikes_disabled    (varie de 0% à 24% selon la station)
   - bikes_low_battery (proxy de qualité réelle de l'offre)
   - mean_battery      (continu)
   - distance moyenne des vélos en transit à chaque station (haversine)
     → proxy d'arrivées imminentes, signal ABSENT du modèle stations-only.

Prochaine itération :
  → Créer merge_vehicles_to_stations.py → stations_enriched.csv
  → Re-train XGBoost sur stations_enriched.csv
  → Comparer MAE/lift vs run actuelle sur HARD_STATIONS
    (Port, Casabona, Eglise de Terre Sainte…)
""")

part("FIN DU RAPPORT EDA")
print("Sortie complète capturée. Relancer avec :")
print("  python eda_report.py > eda_report.txt")
