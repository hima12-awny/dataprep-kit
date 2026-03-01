"""
Top metrics bar displaying key dataset health indicators.
"""

import streamlit as st
from typing import Dict

from core.state import StateManager
from config.theme import theme


def render_metrics_bar():
    """Render a top metrics bar with key dataset stats."""
    dataset = StateManager.get_dataset()

    if dataset is None:
        return

    overview = dataset.overview()

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("🔢 Rows", f"{overview['row_count']:,}")
    col2.metric("📊 Columns", f"{overview['column_count']}")
    col3.metric("🕳️ Missing", f"{overview['missing_percentage']:.1f}%")
    col4.metric("👥 Duplicates", f"{overview['duplicate_rows']:,}")
    col5.metric("💾 Memory", f"{overview['memory_usage_mb']:.5f} MB")