"""Entrypoint Streamlit — navigation entre les 3 pages.

Lancement :
    streamlit run dashboard/app/dashboard.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from views import page_t0, page_pred, page_monit
from data import get_pipeline_health
from components import format_freshness, PRIMARY, MUTED
from glossary import glossary_expander

st.set_page_config(
    page_title="AlterVélo — Dashboard",
    page_icon=":bike:",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = {
    "État actuel (T-0)": ("🚲", page_t0.render),
    "Prévision (T+h)":   ("🔮", page_pred.render),
    "Monitoring":        ("📊", page_monit.render),
}


def _sidebar_health() -> None:
    """Bloc santé pipeline en sidebar — composants Streamlit natifs."""
    try:
        h = get_pipeline_health()
    except Exception as e:
        st.sidebar.error(f"DB inaccessible : {e}")
        return
    obs_txt, obs_col = format_freshness(h["last_obs"])
    pred_txt, pred_col = format_freshness(h["last_pred"])
    with st.sidebar.container(border=True):
        st.caption("SANTÉ PIPELINE")
        st.markdown(
            f'<span style="color:{obs_col};font-weight:700;">●</span> '
            f'Dernière obs : {obs_txt}',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<span style="color:{pred_col};font-weight:700;">●</span> '
            f'Dernière pred : {pred_txt}',
            unsafe_allow_html=True,
        )
        st.caption(
            f"{h['n_obs']:,} obs · {h['n_pred_live']:,} preds live · "
            f"{h['n_pred_back']:,} backtest"
        )


def main():
    st.sidebar.markdown(
        f'<span style="font-size:1.4rem;font-weight:700;color:{PRIMARY};">🚲 AlterVélo Réunion</span>',
        unsafe_allow_html=True,
    )
    st.sidebar.caption("Dashboard de prévision des stations")

    labels = [f"{icon}  {name}" for name, (icon, _) in PAGES.items()]
    choice = st.sidebar.radio("Navigation", labels, label_visibility="collapsed")
    page_name = choice.split("  ", 1)[1]

    st.sidebar.divider()
    _sidebar_health()

    with st.sidebar:
        glossary_expander()

    st.sidebar.divider()
    st.sidebar.caption(
        "Backend : SQLite (`velos.db`)\n\n"
        "Modèle : XGBoost v3 (Δy) — 4 horizons : 60, 90, 120, 150 min\n\n"
        "Pipeline : `dashboard/ingest/pipeline.sh` (cron 30 min)"
    )

    PAGES[page_name][1]()


if __name__ == "__main__":
    main()
