"""
Side-by-side before/after dataframe preview.
Supports full-width and compact (inline) modes.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Optional


def render_preview_table(preview_data: Dict, compact: bool = False):
    """
    Render a before/after comparison from an action preview.

    Args:
        preview_data: Dict with before/after DataFrames and metadata.
        compact: If True, renders in a more compact layout suitable for
                 inline display below an action card.
    """
    if preview_data.get("status") == "error":
        st.error(f"❌ Preview failed: {preview_data.get('error')}")
        return

    if not compact:
        st.markdown("---")
        st.markdown("### 👁️ Action Preview")

    summary = preview_data.get("summary", "")
    if summary:
        st.info(f"📋 {summary}")

    # ── Metrics row ───────────────────────────────────────────
    rows_before = preview_data.get("rows_before", 0)
    rows_after = preview_data.get("rows_after", 0)
    cols_before = preview_data.get("cols_before", 0)
    cols_after = preview_data.get("cols_after", 0)
    affected = preview_data.get("affected_columns", [])

    row_delta = rows_after - rows_before
    col_delta = cols_after - cols_before

    if compact:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{rows_before:,} → {rows_after:,}",
                  delta=f"{row_delta:,}" if row_delta != 0 else None)
        c2.metric("Columns", f"{cols_before} → {cols_after}",
                  delta=f"{col_delta}" if col_delta != 0 else None)
        c3.metric("Affected", len(affected))
        c4.metric("Status", "✅ Valid")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows Before", f"{rows_before:,}")
        col2.metric("Rows After", f"{rows_after:,}",
                    delta=f"{row_delta:,}" if row_delta != 0 else None)
        col3.metric("Columns After", f"{cols_after}",
                    delta=f"{col_delta}" if col_delta != 0 else None)
        col4.metric("Affected Columns", len(affected))

    # ── Before / After tabs ───────────────────────────────────
    before_df = preview_data.get("before")
    after_df = preview_data.get("after")

    if compact:
        max_height = 300
        min_height = 150
    else:
        max_height = 450
        min_height = 200

    tab_before, tab_after, tab_diff = st.tabs(
        ["📊 Before", "📊 After", "🔍 Changes"])

    with tab_before:
        if isinstance(before_df, pd.DataFrame) and not before_df.empty:
            st.dataframe(
                before_df,
                width='stretch',
                height=min(max_height, max(
                    min_height, len(before_df) * 35 + 50)),
            )
            st.caption(f"{len(before_df)} rows shown")
        else:
            st.write("No data available")

    with tab_after:
        if isinstance(after_df, pd.DataFrame) and not after_df.empty:
            st.dataframe(
                after_df,
                width='stretch',
                height=min(max_height, max(
                    min_height, len(after_df) * 35 + 50)),
            )
            st.caption(f"{len(after_df)} rows shown")
        else:
            st.write("No data available")

    with tab_diff:
        _render_diff_summary(preview_data, compact)

    if not compact:
        st.markdown("---")


def _render_diff_summary(preview_data: Dict, compact: bool = False):
    """Render a summary of differences."""
    affected = preview_data.get("affected_columns", [])
    rows_before = preview_data.get("rows_before", 0)
    rows_after = preview_data.get("rows_after", 0)

    if affected:
        st.markdown("**Affected columns:**")
        if compact and len(affected) > 5:
            pills = "  ".join([f"`{col}`" for col in affected])
            st.markdown(pills)
        else:
            for col in affected:
                st.markdown(f"  - `{col}`")

    before_df = preview_data.get("before")
    after_df = preview_data.get("after")

    if isinstance(before_df, pd.DataFrame) and isinstance(after_df, pd.DataFrame):
        common_cols = [
            c for c in affected if c in before_df.columns and c in after_df.columns]
        if common_cols:
            st.markdown("**Null count changes:**")
            diff_data = []
            for col in common_cols:
                nulls_before = int(before_df[col].isna().sum())
                nulls_after = int(after_df[col].isna().sum())
                if nulls_before != nulls_after:
                    diff_data.append({
                        "Column": col,
                        "Nulls Before": nulls_before,
                        "Nulls After": nulls_after,
                        "Change": nulls_after - nulls_before,
                    })
            if diff_data:
                st.dataframe(
                    pd.DataFrame(diff_data),
                    width='stretch',
                    hide_index=True,
                )

        new_cols = [
            c for c in affected if c not in before_df.columns and c in after_df.columns]
        removed_cols = [
            c for c in affected if c in before_df.columns and c not in after_df.columns]

        if new_cols:
            st.markdown(
                f"**New columns added:** {', '.join(f'`{c}`' for c in new_cols)}")
        if removed_cols:
            st.markdown(
                f"**Columns removed:** {', '.join(f'`{c}`' for c in removed_cols)}")

    if rows_before != rows_after:
        diff = rows_after - rows_before
        direction = "added" if diff > 0 else "removed"
        st.markdown(f"**Rows {direction}:** {abs(diff):,}")

    if not affected:
        st.info("No visible changes detected in the preview sample.")


def render_simple_preview(df: pd.DataFrame, title: str = "Data Preview", max_rows: int = 50):
    """Render a simple dataframe preview."""
    st.markdown(f"**{title}** ({len(df):,} rows × {len(df.columns)} columns)")
    st.dataframe(
        df.head(max_rows),
        width='stretch',
        height=min(450, max(200, min(len(df), max_rows) * 35 + 50)),
    )
