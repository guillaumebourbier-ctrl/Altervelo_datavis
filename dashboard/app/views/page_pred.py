"""Page Prévision : projection à T+horizon (60/90/120/150 min)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from data import get_latest_predictions, get_station_volatility, get_monitoring_global
from components import (
    render_station_map, section_header, kpi_card,
    legend_inline, freshness_badge,
    ALERT_RGBA, ALERT_HEX, ALERTS,
)

K_VOLATILITY = 1.5


def classify_alert(y_pred: float, y_current: float, capacity: int,
                   volatility: float) -> str:
    if y_pred <= 1:
        return "rupture"
    if y_pred >= capacity - 2:
        return "saturation"
    if y_pred <= 2 or abs(y_pred - y_current) >= K_VOLATILITY * volatility:
        return "tension"
    return "normal"


def _confidence_dots(sigma: float) -> str:
    if sigma < 1.5:
        return "●●●"
    if sigma < 3.0:
        return "●●○"
    return "●○○"


def render() -> None:
    section_header(
        "Prévision",
        "Projection à T+h via XGBoost v3, comparée à la persistance.",
        term="horizon",
    )

    horizon = st.segmented_control(
        "Horizon de prédiction",
        options=[60, 90, 120, 150],
        format_func=lambda h: f"T + {h} min",
        default=60,
    )
    if horizon is None:
        horizon = 60

    ts_pred, df = get_latest_predictions(horizon)
    if df.empty:
        st.warning(
            f"Aucune prédiction live à horizon {horizon} min. "
            "Lance le pipeline (`bash ingest/pipeline.sh`)."
        )
        return

    df["delta"] = df["y_pred"] - df["y_current"]
    volat = get_station_volatility()
    df["volatility"] = df["station_index"].map(volat).fillna(1.0)
    df["alert"] = [
        classify_alert(p, c, cap, v)
        for p, c, cap, v in zip(
            df["y_pred"], df["y_current"], df["capacity"], df["volatility"]
        )
    ]
    df["color"] = df["alert"].map(ALERT_RGBA)
    df["confiance"] = df["volatility"].apply(_confidence_dots)

    ts_target = pd.to_datetime(ts_pred) + pd.Timedelta(minutes=horizon)
    freshness_badge(ts_pred, label=f"Émise à {ts_pred} · cible {ts_target.isoformat()}")
    st.write("")

    n_rupt = int((df["alert"] == "rupture").sum())
    n_tens = int((df["alert"] == "tension").sum())
    n_satu = int((df["alert"] == "saturation").sum())
    n_total_pred = int(df["y_pred"].round().sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card(f"Vélos prédits T+{horizon}m", n_total_pred,
                 hint="somme arrondie sur le réseau")
    with c2:
        kpi_card("Rupture", n_rupt, term="rupture", accent=ALERT_HEX["rupture"],
                 hint="y_pred ≤ 1")
    with c3:
        kpi_card("Tension", n_tens, term="tension", accent=ALERT_HEX["tension"],
                 hint="y_pred ≤ 2 ou |Δ| ≥ 1.5·σ")
    with c4:
        kpi_card("Saturation", n_satu, term="saturation", accent=ALERT_HEX["saturation"],
                 hint="y_pred ≥ cap − 2")

    st.write("")
    section_header("Carte projetée", f"Statut prédit pour chaque station à T+{horizon} min.")
    tooltip = ("<b>{station_name}</b><br/>"
               "T-0 : {y_current} vélos<br/>"
               f"T+{horizon} m : "
               "{y_pred} vélos (Δ {delta})<br/>"
               "Statut : <b>{alert}</b>")
    render_station_map(df, tooltip_html=tooltip)

    legend_inline([
        (ALERT_HEX["rupture"],    ALERTS["rupture"]["label"],    "rupture"),
        (ALERT_HEX["tension"],    ALERTS["tension"]["label"],    "tension"),
        (ALERT_HEX["saturation"], ALERTS["saturation"]["label"], "saturation"),
        (ALERT_HEX["normal"],     ALERTS["normal"]["label"],     None),
    ])

    st.write("")
    section_header("Détail par station", "Tri par sévérité d'alerte.")
    show = df[[
        "station_name", "y_current", "y_pred", "delta", "capacity", "alert", "confiance",
    ]].sort_values("alert")
    show = show.rename(columns={
        "station_name": "Station",
        "y_current": "T-0",
        "y_pred": f"T+{horizon}m",
        "delta": "Δ",
        "capacity": "Capacité",
        "alert": "Statut",
        "confiance": "Confiance",
    })
    st.dataframe(
        show.style.format({"T-0": "{:.0f}", f"T+{horizon}m": "{:.1f}", "Δ": "{:+.1f}"}),
        use_container_width=True, hide_index=True,
    )

    try:
        kpi = get_monitoring_global(horizon)
        if kpi.get("n"):
            verdict = "bat" if kpi["lift_pct"] >= 0 else "perd contre"
            st.caption(
                f"Sur l'historique évalué ({kpi['n']:,} preds), le modèle {verdict} "
                f"la persistance avec un lift de {kpi['lift_pct']:+.1f}% à {horizon} min. "
                f"Voir l'onglet Monitoring pour le détail."
            )
    except Exception:
        pass
