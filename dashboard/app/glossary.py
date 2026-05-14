"""Définitions des termes métier affichés dans le dashboard.

Source unique de vérité : chaque popover d'aide lit ce dictionnaire.
"""
from __future__ import annotations

import streamlit as st

TERMS: dict[str, dict] = {
    "snapshot": {
        "label": "Snapshot",
        "short": "Photo instantanée du réseau à un timestamp donné (toutes les 30 min).",
        "long": "Un snapshot est une ligne par station collectée via l'API GBFS et "
                "quantizée à la grille 30 min. Toutes les pages T-0 affichent le snapshot le plus récent.",
    },
    "horizon": {
        "label": "Horizon",
        "short": "Délai entre l'instant de prédiction (T-0) et la cible (T+h).",
        "long": "Le modèle XGBoost prédit l'état des stations à T+60, +90, +120 et +150 min. "
                "Plus l'horizon est grand, plus l'incertitude croît.",
    },
    "delta": {
        "label": "Δ (delta)",
        "short": "Écart prédit entre T-0 et T+h : Δ = y_pred − y_current.",
        "long": "Le modèle est entraîné à prédire Δy (et non la valeur absolue), puis on rajoute "
                "y_current pour obtenir y_pred. Un Δ positif = la station va se remplir.",
    },
    "mae": {
        "label": "MAE (Mean Absolute Error)",
        "short": "Erreur absolue moyenne — en nombre de vélos.",
        "long": "MAE = moyenne sur toutes les prédictions évaluées de |y_pred − y_obs|. "
                "Une MAE de 0.1 = en moyenne, le modèle se trompe de 0.1 vélo par station. "
                "Plus c'est bas, mieux c'est.",
        "formula": r"\text{MAE} = \frac{1}{N}\sum_{i=1}^{N} |y^{pred}_i - y^{obs}_i|",
    },
    "persistance": {
        "label": "Persistance (baseline)",
        "short": "Prédiction triviale : « ça reste pareil » (y_pred = y_current).",
        "long": "C'est la baseline contre laquelle le modèle doit se comparer. "
                "Sur des séries quasi-stationnaires (vélos peu utilisés la nuit), "
                "la persistance est très dure à battre.",
    },
    "lift": {
        "label": "Lift",
        "short": "Gain relatif du modèle vs persistance, en %.",
        "long": "Lift = (MAE_persistance − MAE_modèle) / MAE_persistance × 100. "
                "Lift positif = le modèle bat la baseline. Lift négatif = la persistance gagne.",
        "formula": r"\text{Lift} = \frac{\text{MAE}_{pers} - \text{MAE}_{model}}{\text{MAE}_{pers}} \times 100",
    },
    "hit_rate": {
        "label": "Hit rate ±k",
        "short": "Part des prédictions à moins de k vélos d'erreur.",
        "long": "Hit rate ±1 = part des prédictions où |y_pred − y_obs| ≤ 1. "
                "Plus parlant que la MAE pour un opérateur : « 9 fois sur 10, je tombe à 1 vélo près ».",
    },
    "volatility": {
        "label": "Volatilité σ",
        "short": "Écart-type historique du nombre de vélos par station.",
        "long": "Calculé sur tout l'historique d'observations. Un hub (Port, Casabona) a une σ ≈ 4, "
                "une petite station a σ ≈ 0.8. Sert à classifier l'alerte « tension » de manière adaptative : "
                "un Δ de 3 est énorme pour une petite station mais normal pour un hub.",
    },
    "rupture": {
        "label": "Alerte rupture",
        "short": "Station avec ≤ 1 vélo prédit — réapprovisionnement urgent.",
        "long": "Critère : y_pred ≤ 1. L'opérateur doit envoyer une benne dans l'heure.",
    },
    "tension": {
        "label": "Alerte tension",
        "short": "Station à risque — peu de vélos OU mouvement anormal.",
        "long": "Critère : y_pred ≤ 2 OU |Δ| ≥ 1.5 × σ_station. Dans les deux cas, "
                "la station mérite une surveillance accrue.",
    },
    "saturation": {
        "label": "Alerte saturation",
        "short": "Station presque pleine — plus de docks pour rendre un vélo.",
        "long": "Critère : y_pred ≥ capacité − 2. Les usagers vont chercher à rendre ailleurs.",
    },
    "backtest": {
        "label": "Backtest",
        "short": "Prédictions générées sur l'historique (test set 20 %).",
        "long": "Le bootstrap (db_init.py) rejoue le modèle sur les 20 % de test pour "
                "constituer une base d'évaluation immédiate (~30 000 prédictions). "
                "Source = 'backtest' dans la table predictions.",
    },
    "live": {
        "label": "Live",
        "short": "Prédictions générées par le tick cron toutes les 30 min.",
        "long": "Source = 'live' dans la table predictions. Y_obs et erreurs sont remplies "
                "en différé (delayed-feedback) dès que ts_target est observé.",
    },
    "trace": {
        "label": "Trajectoire (trace)",
        "short": "Comparaison visuelle pour une station : observation, persistance, modèle.",
        "long": "Pour un horizon h donné, on aligne sur le même axe ts_target trois courbes : "
                "ce qu'on a observé (y_obs), ce que la persistance prédisait (= y_current émis à T-h), "
                "et ce que le modèle prédisait (y_pred). La persistance ressemble à l'observation "
                "décalée de h minutes ; le modèle, lui, doit anticiper.",
    },
    "mae_arrondie": {
        "label": "MAE arrondie (lecture honnête)",
        "short": "MAE recalculée après arrondi du modèle à l'entier — comparable à la persistance.",
        "long": "La MAE brute compare un entier (persistance, |y_obs − y_current|) à un nombre continu "
                "(modèle, |y_obs − y_pred|), ce qui pénalise le modèle de ~0.5 vélo par fraction même "
                "quand il vise la bonne classe. La MAE arrondie remplace y_pred par round(y_pred) pour "
                "remettre les deux prédicteurs sur la même échelle entière. À utiliser comme contrepoint "
                "pour évaluer la valeur opérationnelle réelle — pas comme métrique de remplacement, "
                "car l'arrondi masque la nuance continue (intervalles, tendances) que le modèle apporte.",
        "formula": r"\text{MAE}_{round} = \frac{1}{N}\sum_{i=1}^{N} |y^{obs}_i - \mathrm{round}(y^{pred}_i)|",
    },
    "imputation": {
        "label": "Imputation",
        "short": "Comblage des snapshots manquants (panne API, etc.).",
        "long": "Stratégie : ffill la nuit (peu de mouvement), interpolation linéaire le jour. "
                "Marqué is_imputed=1 dans stations_clean.",
    },
}


def render_term(term: str) -> None:
    """Rend une définition complète (court + long + LaTeX) dans un popover."""
    t = TERMS.get(term)
    if t is None:
        st.write(f"_(Terme inconnu : `{term}`)_")
        return
    st.markdown(f"**{t['label']}**")
    st.caption(t["short"])
    st.markdown(t["long"])
    if "formula" in t:
        st.latex(t["formula"])


def help_badge(term: str, label: str = "?") -> None:
    """Petit popover '?' à placer à côté d'un titre ou d'une métrique."""
    if term not in TERMS:
        return
    with st.popover(label, use_container_width=False):
        render_term(term)


def glossary_expander() -> None:
    """Expander complet listant tous les termes — pour la sidebar."""
    with st.expander("Glossaire"):
        for term in TERMS:
            with st.container():
                render_term(term)
                st.divider()
