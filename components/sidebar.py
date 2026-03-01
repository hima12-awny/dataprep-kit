# """
# Global sidebar: dataset health indicators, navigation context, undo/redo.
# """

# import streamlit as st
# from typing import Optional

# from core.state import StateManager
# from config.theme import theme


# def render_sidebar():
#     """Render the global sidebar with dataset health and controls."""
#     st.sidebar.markdown(theme.get_custom_css(), unsafe_allow_html=True)

#     st.sidebar.markdown("## 🧪 DataPrep Kit")
#     st.sidebar.markdown("---")

#     dataset = StateManager.get_dataset()

#     if dataset is None:
#         st.sidebar.info(
#             "📂 No dataset loaded. Go to **Import** page to get started.")
#         return

#     # ── Dataset Info ──────────────────────────────────────────
#     st.sidebar.markdown("### 📊 Dataset")
#     st.sidebar.text(f"Source: {dataset.source_name}")

#     overview = dataset.overview()

#     col1, col2 = st.sidebar.columns(2)
#     col1.metric("Rows", f"{overview['row_count']:,}")
#     col2.metric("Columns", f"{overview['column_count']}")

#     col1, col2 = st.sidebar.columns(2)
#     col1.metric("Missing %", f"{overview['missing_percentage']:.1f}%")
#     col2.metric("Duplicates", f"{overview['duplicate_rows']:,}")

#     st.sidebar.metric("Memory", f"{overview['memory_usage_mb']:.1f} MB")

#     # ── Data Quality Indicator ────────────────────────────────
#     quality_score = _compute_quality_score(overview)
#     _render_quality_bar(quality_score)

#     st.sidebar.markdown("---")

#     # ── Undo / Redo ───────────────────────────────────────────
#     st.sidebar.markdown("### ↩️ History")
#     col1, col2 = st.sidebar.columns(2)

#     with col1:
#         if st.button(
#             f"↩ Undo ({dataset.undo_steps_count})",
#             disabled=not dataset.can_undo,
#             width='stretch',
#         ):
#             if dataset.undo():
#                 StateManager.invalidate_profiling_cache()
#                 StateManager.add_notification("Undo successful", "success")
#                 st.rerun()

#     with col2:
#         if st.button(
#             "↪ Redo",
#             disabled=not dataset.can_redo,
#             width='stretch',
#         ):
#             if dataset.redo():
#                 StateManager.invalidate_profiling_cache()
#                 StateManager.add_notification("Redo successful", "success")
#                 st.rerun()

#     # ── Pipeline Summary ──────────────────────────────────────
#     pipeline = StateManager.get_pipeline()
#     st.sidebar.markdown("---")
#     st.sidebar.markdown("### 📋 Pipeline")
#     st.sidebar.text(f"Steps: {pipeline.step_count}")
#     st.sidebar.text(f"Enabled: {len(pipeline.enabled_steps)}")

#     # ── Column Types Summary ──────────────────────────────────
#     st.sidebar.markdown("---")
#     st.sidebar.markdown("### 🏷️ Column Types")
#     dtypes = overview.get("dtypes_summary", {})
#     for dtype, count in dtypes.items():
#         st.sidebar.text(f"  {dtype}: {count}")


# def _compute_quality_score(overview: dict) -> float:
#     """Compute a simple data quality score (0–100)."""
#     score = 100.0

#     # Penalize missing data
#     missing_pct = overview.get("missing_percentage", 0)
#     score -= min(missing_pct * 1.5, 40)

#     # Penalize duplicates
#     dup_pct = overview.get("duplicate_percentage", 0)
#     score -= min(dup_pct * 2, 20)

#     return max(0, round(score, 1))


# def _render_quality_bar(score: float):
#     """Render a colored quality score bar."""
#     if score >= 80:
#         color = theme.SUCCESS
#         label = "Good"
#     elif score >= 60:
#         color = theme.WARNING
#         label = "Fair"
#     else:
#         color = theme.DANGER
#         label = "Poor"

#     st.sidebar.markdown(
#         f"""
#         <div style="margin: 0.5rem 0;">
#             <div style="display: flex; justify-content: space-between; font-size: 0.85rem;">
#                 <span>Data Quality</span>
#                 <span style="color: {color}; font-weight: 600;">{score:.0f}/100 ({label})</span>
#             </div>
#             <div style="background: #e5e7eb; border-radius: 4px; height: 8px; margin-top: 4px;">
#                 <div style="background: {color}; width: {score}%; height: 100%;
#                      border-radius: 4px; transition: width 0.5s;"></div>
#             </div>
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )


"""
Sidebar: dataset health indicators, undo/redo, pipeline summary.
Navigation is handled by st.navigation — this only adds health info.
"""

import streamlit as st

from core.state import StateManager
from config.theme import theme


def render_sidebar():
    """Render dataset health info in the sidebar (below navigation)."""
    dataset = StateManager.get_dataset()

    if dataset is None:
        st.sidebar.info("📂 No dataset loaded. Go to **Import Data**.")
        return

    st.sidebar.markdown("---")

    # ── Dataset Info ──────────────────────────────────────────
    overview = dataset.overview()

    st.sidebar.markdown(f"**📊 {dataset.source_name}**")

    c1, c2 = st.sidebar.columns(2)
    c1.metric("Rows", f"{overview['row_count']:,}")
    c2.metric("Cols", f"{overview['column_count']}")

    c1, c2 = st.sidebar.columns(2)
    c1.metric("Missing", f"{overview['missing_percentage']:.1f}%")
    c2.metric("Dups", f"{overview['duplicate_rows']:,}")

    # ── Quality Bar ───────────────────────────────────────────
    score = 100.0 - min(overview.get("missing_percentage", 0) * 1.5, 40) - \
        min(overview.get("duplicate_percentage", 0) * 2, 20)
    score = max(0, round(score, 1))
    color = "#10B981" if score >= 80 else "#F59E0B" if score >= 60 else "#EF4444"
    label = "Good" if score >= 80 else "Fair" if score >= 60 else "Poor"

    st.sidebar.markdown(
        f"""<div style="font-size:0.85rem;display:flex;justify-content:space-between;">
        <span>Quality</span><span style="color:{color};font-weight:600;">{score:.0f}/100 ({label})</span></div>
        <div style="background:#374151;border-radius:4px;height:6px;margin-top:4px;">
        <div style="background:{color};width:{score}%;height:100%;border-radius:4px;"></div></div>""",
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")

    # ── Undo / Redo ───────────────────────────────────────────
    c1, c2 = st.sidebar.columns(2)
    with c1:
        if st.button(f"↩ Undo ({dataset.undo_steps_count})", disabled=not dataset.can_undo, width='content'):
            dataset.undo()
            StateManager.invalidate_profiling_cache()
            st.rerun()
    with c2:
        if st.button("↪ Redo", disabled=not dataset.can_redo, width='content'):
            dataset.redo()
            StateManager.invalidate_profiling_cache()
            st.rerun()

    # ── Pipeline & AI Status ──────────────────────────────────
    pipeline = StateManager.get_pipeline()
    st.sidebar.caption(f"📋 Pipeline: {pipeline.step_count} steps")

    ai_config = StateManager.get_ai_config()
    if ai_config.get("api_key"):
        st.sidebar.caption("🤖 AI: Configured ✅")
    else:
        st.sidebar.caption("🤖 AI: Not configured")
