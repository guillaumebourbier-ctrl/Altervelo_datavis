"""Page T-0 : état actuel du réseau au dernier snapshot disponible."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from data import get_latest_snapshot
from components import (
    render_station_map, section_header, kpi_card,
    legend_inline, freshness_badge, progress_capacity,
    classify_state, ALERT_RGBA, ALERT_HEX, ALERTS,
)


def render() -> None:
    section_header(
        "État du réseau",
        "Photo instantanée du dernier snapshot GBFS quantizé à 30 min.",
        term="snapshot",
    )

    latest_ts, df = get_latest_snapshot()
    if df.empty:
        st.error("Aucune observation disponible. Lance `db_init.py` d'abord.")
        return

    df["fill_ratio"] = df["num_vehicles_available"] / df["capacity"].replace(0, pd.NA)
    df["alert"] = [
        classify_state(y, c)
        for y, c in zip(df["num_vehicles_available"], df["capacity"])
    ]
    df["color"] = df["alert"].map(ALERT_RGBA)

    n_stations = len(df)
    n_total = int(df["num_vehicles_available"].sum())
    n_rupt = int((df["alert"] == "rupture").sum())
    n_tens = int((df["alert"] == "tension").sum())
    n_satu = int((df["alert"] == "saturation").sum())

    freshness_badge(latest_ts, label=f"Snapshot {latest_ts}")
    st.write("")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Vélos en circulation", n_total, hint=f"sur {n_stations} stations")
    with c2:
        kpi_card("Rupture", n_rupt, term="rupture",
                 accent=ALERT_HEX["rupture"], hint="≤ 1 vélo")
    with c3:
        kpi_card("Tension", n_tens, term="tension",
                 accent=ALERT_HEX["tension"], hint="2 vélos ou moins")
    with c4:
        kpi_card("Saturation", n_satu, term="saturation",
                 accent=ALERT_HEX["saturation"], hint="≥ capacité − 2")

    st.write("")
    section_header("Carte du réseau",
                   "Couleur = statut courant de la station (mêmes seuils que la prévision).")
    tooltip = ("<b>{station_name}</b><br/>"
               "Vélos : {num_vehicles_available} / {capacity}<br/>"
               "Statut : <b>{alert}</b>")
    render_station_map(df, tooltip_html=tooltip)

    legend_inline([
        (ALERT_HEX["rupture"],    ALERTS["rupture"]["label"],    "rupture"),
        (ALERT_HEX["tension"],    ALERTS["tension"]["label"],    "tension"),
        (ALERT_HEX["saturation"], ALERTS["saturation"]["label"], "saturation"),
        (ALERT_HEX["normal"],     ALERTS["normal"]["label"],     None),
    ])

    st.write("")
    section_header("Détail par station")
    df["remplissage"] = df.apply(
        lambda r: progress_capacity(r["num_vehicles_available"], r["capacity"]), axis=1
    )
    show = df[[
        "station_name", "remplissage", "fill_ratio", "alert",
        "n_vehicles_actifs", "mean_battery", "is_obs_missing",
    ]].rename(columns={
        "station_name": "Station",
        "remplissage": "Vélos / Capacité",
        "fill_ratio": "%",
        "alert": "Statut",
        "n_vehicles_actifs": "Actifs (API)",
        "mean_battery": "Batterie moy.",
        "is_obs_missing": "Sans obs",
    })
    st.dataframe(
        show.style.format({"%": "{:.0%}", "Batterie moy.": "{:.0%}"}),
        use_container_width=True,
        hide_index=True,
    )
