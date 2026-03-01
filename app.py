"""
DataPrep Kit — Main entry point with st.navigation.
Run with: streamlit run app.py
"""

import streamlit as st

from core.state import StateManager
from config.settings import settings
from config.theme import theme
from components.sidebar import render_sidebar


# ── Page Config (must be first Streamlit call) ────────────────
st.set_page_config(
    page_title=settings.APP_NAME,
    page_icon=settings.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Initialize State ──────────────────────────────────────────
StateManager.initialize()

# ── Apply Custom CSS ──────────────────────────────────────────
st.markdown(theme.get_custom_css(), unsafe_allow_html=True)
try:
    with open("assets/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# ── Navigation ────────────────────────────────────────────────
pg = st.navigation(
    {
        "Overview": [
            st.Page("pages/home_page.py", title="Home",
                    icon=":material/home:"),
        ],
        "Data Preparation": [
            st.Page("pages/import_page.py", title="Import Data",
                    icon=":material/upload_file:"),
            st.Page("pages/profiling_page.py", title="Profiling",
                    icon=":material/analytics:"),
            st.Page("pages/cleaning_page.py", title="Cleaning",
                    icon=":material/cleaning_services:"),
            st.Page("pages/conversion_page.py", title="Conversion",
                    icon=":material/sync_alt:"),
            st.Page("pages/engineering_page.py",
                    title="Feature Engineering", icon=":material/construction:"),
        ],
        "Pipeline & Export": [
            st.Page("pages/pipeline_page.py", title="Pipeline",
                    icon=":material/account_tree:"),
            st.Page("pages/export_page.py", title="Export",
                    icon=":material/download:"),
        ],
        "Configuration": [
            st.Page("pages/ai_settings_page.py", title="AI Settings",
                    icon=":material/smart_toy:"),
        ],
    }
)

# ── Display Notifications ─────────────────────────────────────
for notif in StateManager.get_notifications():
    level = notif.get("level", "info")
    msg = notif.get("message", "")
    getattr(st, level if level in (
        "success", "warning", "error", "info") else "info")(msg)

# ── Sidebar Health Info ───────────────────────────────────────
render_sidebar()

# ── Run Selected Page ─────────────────────────────────────────
pg.run()
