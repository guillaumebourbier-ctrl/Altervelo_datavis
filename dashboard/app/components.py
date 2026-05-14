"""Composants UI partagés : carte pydeck, barème couleur, KPI cards, en-têtes."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pydeck as pdk
import streamlit as st

from glossary import help_badge

REUNION_VIEW = pdk.ViewState(latitude=-21.115, longitude=55.536, zoom=10, pitch=0)

PRIMARY = "#1f9d55"
MUTED = "#6b7280"
DANGER = "#dc2626"
WARN = "#d97706"
OK = "#16a34a"

# Source unique de vérité pour les couleurs d'alerte (utilisées sur T-0 et Prévision).
ALERTS = {
    "rupture":    {"hex": "#dc2626", "rgba": [220,  38,  38, 220], "label": "Rupture"},
    "tension":    {"hex": "#d97706", "rgba": [217, 119,   6, 220], "label": "Tension"},
    "saturation": {"hex": "#7c3aed", "rgba": [124,  58, 237, 220], "label": "Saturation"},
    "normal":     {"hex": "#16a34a", "rgba": [ 22, 163,  74, 200], "label": "Normal"},
}
ALERT_RGBA = {k: v["rgba"] for k, v in ALERTS.items()}
ALERT_HEX  = {k: v["hex"]  for k, v in ALERTS.items()}


def classify_state(y: float, capacity: int) -> str:
    """Classification d'état basée sur le nombre de vélos et la capacité.

    Mêmes seuils que la prévision mais sans la composante volatilité (pas de Δ).
    """
    if y is None or pd.isna(y):
        return "normal"
    if y <= 1:
        return "rupture"
    if y >= capacity - 2:
        return "saturation"
    if y <= 2:
        return "tension"
    return "normal"


def color_fill_ratio(ratio: float) -> list[int]:
    """0 (vide) → rouge ; 0.5 → orange ; 1 (plein) → vert. RGBA."""
    if ratio is None or pd.isna(ratio):
        return [120, 120, 120, 200]
    r = max(0.0, min(1.0, float(ratio)))
    if r < 0.5:
        red, green = 230, int(120 * (r / 0.5))
    else:
        red, green = int(230 - 200 * ((r - 0.5) / 0.5)), int(120 + 80 * ((r - 0.5) / 0.5))
    return [red, green, 30, 200]


def color_mae(mae: float, max_mae: float = 1.0) -> list[int]:
    """0 (parfait) → vert ; >= max_mae → rouge."""
    if mae is None or pd.isna(mae):
        return [120, 120, 120, 200]
    r = min(1.0, mae / max(max_mae, 1e-6))
    return [int(230 * r), int(180 * (1 - r)), 30, 200]


def render_station_map(df: pd.DataFrame, color_col: str = "color",
                       tooltip_html: str | None = None,
                       radius_col: str | None = None,
                       height: int = 520) -> None:
    """Carte ScatterplotLayer générique. df doit contenir lat, lon et color_col (RGBA list)."""
    if df.empty:
        st.info("Pas de données à afficher.")
        return
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position="[lon, lat]",
        get_fill_color=color_col,
        get_radius=radius_col if radius_col else 80,
        radius_min_pixels=6,
        radius_max_pixels=30,
        pickable=True,
        stroked=True,
        get_line_color=[40, 40, 40, 200],
        line_width_min_pixels=1,
    )
    tooltip = {"html": tooltip_html} if tooltip_html else {"text": "{station_name}"}
    deck = pdk.Deck(
        map_style="light",
        initial_view_state=REUNION_VIEW,
        layers=[layer],
        tooltip=tooltip,
    )
    st.pydeck_chart(deck, height=height)


# ─── Header / KPI / légendes ─────────────────────────────────────────

def section_header(title: str, subtitle: str | None = None,
                   term: str | None = None) -> None:
    """Titre de section avec popover d'aide optionnel — composants Streamlit natifs."""
    if term:
        c1, c2 = st.columns([20, 1])
        with c1:
            st.subheader(title, anchor=False, divider="green")
        with c2:
            help_badge(term)
    else:
        st.subheader(title, anchor=False, divider="green")
    if subtitle:
        st.caption(subtitle)


def kpi_card(label: str, value, delta: str | None = None,
             term: str | None = None, fmt: str | None = None,
             accent: str | None = None, hint: str | None = None) -> None:
    """KPI card — composants Streamlit natifs + un seul span coloré pour la valeur."""
    if fmt and value is not None and not isinstance(value, str):
        try:
            value_str = format(value, fmt)
        except (TypeError, ValueError):
            value_str = str(value)
    else:
        value_str = "–" if value is None else str(value)

    accent = accent or PRIMARY

    with st.container(border=True):
        if term:
            c1, c2 = st.columns([10, 1])
            with c1:
                st.caption(label.upper())
            with c2:
                help_badge(term)
        else:
            st.caption(label.upper())
        st.markdown(
            f'<span style="font-size:1.9rem;font-weight:700;color:{accent};line-height:1;">{value_str}</span>',
            unsafe_allow_html=True,
        )
        if delta:
            st.caption(delta)
        if hint:
            st.caption(hint)


def legend_inline(items: list[tuple[str, str, str | None]]) -> None:
    """Légende horizontale : list of (couleur_hex, label, term_glossaire_optionnel).

    Utilise un seul span inline par colonne — pas de div imbriqué.
    """
    cols = st.columns(len(items))
    for col, (color, label, term) in zip(cols, items):
        with col:
            if term:
                c1, c2 = st.columns([10, 2])
                with c1:
                    st.markdown(
                        f'<span style="color:{color};font-size:1.1rem;">●</span> '
                        f'<span style="font-size:0.9rem;">{label}</span>',
                        unsafe_allow_html=True,
                    )
                with c2:
                    help_badge(term)
            else:
                st.markdown(
                    f'<span style="color:{color};font-size:1.1rem;">●</span> '
                    f'<span style="font-size:0.9rem;">{label}</span>',
                    unsafe_allow_html=True,
                )


def progress_capacity(value: float, capacity: int) -> str:
    """Format simple « N / capacité » utilisable dans st.dataframe."""
    if capacity <= 0 or value is None or pd.isna(value):
        return "–"
    return f"{int(value)}/{int(capacity)}"


def format_freshness(ts: str | None) -> tuple[str, str]:
    """Retourne (texte, couleur_hex) selon l'âge du timestamp.

    < 35 min : vert, < 90 min : orange, sinon rouge. Aucun ts → gris.
    """
    if ts is None:
        return "Aucune donnée", MUTED
    try:
        dt = pd.to_datetime(ts)
        if dt.tzinfo is None:
            dt = dt.tz_localize("UTC")
        now = datetime.now(timezone.utc)
        age_min = (now - dt.to_pydatetime()).total_seconds() / 60
    except Exception:
        return str(ts), MUTED
    if age_min < 35:
        color = OK
    elif age_min < 90:
        color = WARN
    else:
        color = DANGER
    if age_min < 1:
        txt = "à l'instant"
    elif age_min < 60:
        txt = f"il y a {int(age_min)} min"
    elif age_min < 1440:
        txt = f"il y a {int(age_min / 60)} h"
    else:
        txt = f"il y a {int(age_min / 1440)} j"
    return txt, color


def freshness_badge(ts: str | None, label: str = "Dernière obs") -> None:
    """Affiche un badge fraîcheur (span inline, pas de div)."""
    txt, color = format_freshness(ts)
    st.markdown(
        f'<span style="display:inline-block;padding:4px 10px;border-radius:999px;'
        f'background:{color}22;color:{color};font-size:0.82rem;font-weight:600;">'
        f'● {label} : {txt}</span>',
        unsafe_allow_html=True,
    )
