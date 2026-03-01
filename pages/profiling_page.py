"""
Page 2: Data Profiling
Statistical summaries, distributions, correlations, and quality score.
All data rendered with proper UI widgets — no raw JSON.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from core.state import StateManager
from utils.stats_helpers import StatsHelper
from components.metrics_bar import render_metrics_bar


def main():

    st.markdown(
        """
        <div class="page-header">
            <h1>📊 Data Profiling</h1>
            <p>Explore your dataset's structure, statistics, and quality</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not StateManager.has_dataset():
        st.warning("⚠️ No dataset loaded. Go to the **Import** page first.")
        return

    dataset = StateManager.get_dataset()
    df = dataset.df
    render_metrics_bar()

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Overview", "📊 Distributions", "🔗 Correlations", "🔍 Column Details"
    ])

    with tab1:
        _render_overview(df, dataset)
    with tab2:
        _render_distributions(df)
    with tab3:
        _render_correlations(df)
    with tab4:
        _render_column_details(df, dataset)


# ══════════════════════════════════════════════════════════════
#  Tab 1: Overview
# ══════════════════════════════════════════════════════════════

def _render_overview(df: pd.DataFrame, dataset):
    """Render dataset overview with metric cards instead of JSON."""
    overview = dataset.overview()

    st.markdown("### Dataset Summary")

    # ── Shape & Memory Metrics ────────────────────────────
    st.markdown("##### 📐 Shape & Memory")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{overview['row_count']:,}")
    c2.metric("Columns", f"{overview['column_count']}")
    c3.metric("Total Cells", f"{overview['total_cells']:,}")
    c4.metric("Memory", f"{overview['memory_usage_mb']:.2f} MB")

    # ── Data Quality Metrics ──────────────────────────────
    st.markdown("##### 🩺 Data Quality")
    c1, c2, c3, c4 = st.columns(4)

    missing_pct = overview.get("missing_percentage", 0)
    dup_pct = overview.get("duplicate_percentage", 0)

    # Color-code quality
    if missing_pct == 0:
        missing_icon = "✅"
    elif missing_pct < 5:
        missing_icon = "🟡"
    else:
        missing_icon = "🔴"

    if dup_pct == 0:
        dup_icon = "✅"
    elif dup_pct < 1:
        dup_icon = "🟡"
    else:
        dup_icon = "🔴"

    c1.metric(f"{missing_icon} Total Missing",
              f"{overview['total_missing']:,}")
    c2.metric("Missing %", f"{missing_pct:.1f}%")
    c3.metric(f"{dup_icon} Duplicate Rows", f"{overview['duplicate_rows']:,}")
    c4.metric("Duplicate %", f"{dup_pct:.1f}%")

    # ── Quality Score Bar ─────────────────────────────────
    quality_score = max(0, 100 - missing_pct - dup_pct)
    _render_quality_bar(quality_score)

    # ── Missing value heatmap ─────────────────────────────
    st.markdown("### 🕳️ Missing Values Heatmap")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if len(missing) == 0:
        st.success("✅ No missing values found!")
    else:
        fig = px.bar(
            x=missing.index,
            y=missing.values,
            labels={"x": "Column", "y": "Missing Count"},
            title="Missing Values by Column",
            color=missing.values,
            color_continuous_scale="Reds",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')

    # ── Data types summary ────────────────────────────────
    st.markdown("### 🏷️ Column Types Overview")

    type_df = pd.DataFrame({
        "Column": df.columns,
        "Type": [str(df[col].dtype) for col in df.columns],
        "Non-Null": [df[col].notna().sum() for col in df.columns],
        "Null Count": [df[col].isna().sum() for col in df.columns],
        "Null %": [f"{df[col].isna().mean()*100:.1f}%" for col in df.columns],
        "Unique": [StatsHelper.safe_nunique(df[col]) for col in df.columns],
    })
    st.dataframe(type_df, width='stretch', hide_index=True)

    # ── Type distribution chart ───────────────────────────
    st.markdown("##### 📊 Column Type Distribution")
    type_counts = pd.Series(
        [str(df[col].dtype) for col in df.columns]
    ).value_counts()

    fig = px.pie(
        values=type_counts.values,
        names=type_counts.index,
        title="Column Types",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, width='stretch')


def _render_quality_bar(score: float):
    """Render a visual quality score bar."""
    if score >= 90:
        color = "#10B981"
        label = "Excellent"
    elif score >= 70:
        color = "#F59E0B"
        label = "Good"
    elif score >= 50:
        color = "#F97316"
        label = "Fair"
    else:
        color = "#EF4444"
        label = "Poor"

    st.markdown(
        f"""
        <div style="margin: 0.5rem 0 1rem 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span style="font-weight: 600; font-size: 0.9rem;">🏆 Quality Score</span>
                <span style="font-weight: 700; color: {color};">{score:.0f}% — {label}</span>
            </div>
            <div style="background: #1F2937; border-radius: 8px; height: 12px; overflow: hidden;">
                <div style="
                    width: {min(score, 100)}%;
                    height: 100%;
                    background: {color};
                    border-radius: 8px;
                    transition: width 0.5s ease;
                "></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════
#  Tab 2: Distributions
# ══════════════════════════════════════════════════════════════

def _render_distributions(df: pd.DataFrame):
    """Render distribution plots for numeric and categorical columns."""
    st.markdown("### 📊 Numeric Distributions")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if numeric_cols:
        selected_num = st.selectbox(
            "Select numeric column", numeric_cols, key="dist_num")
        if selected_num:
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(
                    df, x=selected_num, title=f"Histogram: {selected_num}", nbins=50)
                fig.update_layout(height=350)
                st.plotly_chart(fig, width='stretch')
            with col2:
                fig = px.box(df, y=selected_num,
                             title=f"Box Plot: {selected_num}")
                fig.update_layout(height=350)
                st.plotly_chart(fig, width='stretch')
    else:
        st.info("No numeric columns available.")

    st.markdown("### 📊 Categorical Distributions")
    cat_cols = df.select_dtypes(
        include=["object", "category"]).columns.tolist()

    if cat_cols:
        selected_cat = st.selectbox(
            "Select categorical column", cat_cols, key="dist_cat")
        if selected_cat:
            value_counts = df[selected_cat].value_counts().head(20)
            fig = px.bar(
                x=value_counts.index.astype(str),
                y=value_counts.values,
                labels={"x": selected_cat, "y": "Count"},
                title=f"Top 20 values: {selected_cat}",
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch')
    else:
        st.info("No categorical columns available.")


# ══════════════════════════════════════════════════════════════
#  Tab 3: Correlations
# ══════════════════════════════════════════════════════════════

def _render_correlations(df: pd.DataFrame):
    """Render correlation matrix heatmap."""
    st.markdown("### 🔗 Correlation Matrix")

    corr = StatsHelper.correlation_matrix(df)
    if corr is None:
        st.info("Need at least 2 numeric columns for correlation analysis.")
        return

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Correlation Heatmap",
        aspect="auto",
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, width='stretch')

    # High correlation pairs
    from config.settings import settings
    pairs = StatsHelper.high_correlation_pairs(
        df, threshold=settings.CORRELATION_HIGH_THRESHOLD)
    if pairs:
        st.markdown(
            f"### ⚠️ Highly Correlated Pairs (|r| ≥ {settings.CORRELATION_HIGH_THRESHOLD})")
        for col1, col2, corr_val in pairs:
            st.markdown(f"- **{col1}** ↔ **{col2}**: `{corr_val:.4f}`")


# ══════════════════════════════════════════════════════════════
#  Tab 4: Column Details
# ══════════════════════════════════════════════════════════════

def _render_column_details(df: pd.DataFrame, dataset):
    """Render detailed stats for individual columns with proper UI."""
    st.markdown("### 🔍 Column Explorer")

    selected_col = st.selectbox(
        "Select a column", df.columns.tolist(), key="col_explorer")

    if not selected_col:
        return

    stats = dataset.column_stats(selected_col)
    col_dtype = str(df[selected_col].dtype)
    is_numeric = pd.api.types.is_numeric_dtype(df[selected_col])
    is_datetime = pd.api.types.is_datetime64_any_dtype(df[selected_col])
    is_text = col_dtype in ("object", "string")

    # ── Header with type badge ────────────────────────────
    type_color = _dtype_color(col_dtype)
    st.markdown(
        f"#### `{selected_col}` "
        f"<span style='background:{type_color}; color:white; padding:2px 10px; "
        f"border-radius:10px; font-size:0.8rem; font-weight:600;'>{col_dtype}</span>",
        unsafe_allow_html=True,
    )

    # ══════════════════════════════════════════════════════
    #  Row 1: Basic Info Metrics
    # ══════════════════════════════════════════════════════

    st.markdown("##### 📋 Basic Info")

    total = stats.get("total_count", len(df))
    non_null = stats.get("non_null_count", 0)
    null_count = stats.get("null_count", 0)
    null_pct = stats.get("null_percentage", 0)
    unique = stats.get("unique_count", 0)
    unique_pct = stats.get("unique_percentage", 0)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total", f"{total:,}")
    c2.metric("Non-Null", f"{non_null:,}")

    # Color-code null metric
    if null_count == 0:
        c3.metric("✅ Nulls", "0")
    else:
        c3.metric("⚠️ Nulls", f"{null_count:,}",
                  delta=f"{null_pct:.1f}%", delta_color="inverse")

    c4.metric("Unique", f"{unique:,}")
    c5.metric("Unique %", f"{unique_pct:.1f}%")

    # ══════════════════════════════════════════════════════
    #  Row 2: Numeric Statistics (if applicable)
    # ══════════════════════════════════════════════════════

    if is_numeric:
        st.markdown("##### 📊 Numeric Statistics")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean", _fmt_num(stats.get("mean")))
        c2.metric("Median", _fmt_num(stats.get("median")))
        c3.metric("Std Dev", _fmt_num(stats.get("std")))
        c4.metric("IQR", _fmt_num(stats.get("iqr")))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Min", _fmt_num(stats.get("min")))
        c2.metric("Q1 (25%)", _fmt_num(stats.get("q1")))
        c3.metric("Q3 (75%)", _fmt_num(stats.get("q3")))
        c4.metric("Max", _fmt_num(stats.get("max")))

        # Skewness & Kurtosis with interpretation
        c1, c2 = st.columns(2)
        skew = stats.get("skewness")
        kurt = stats.get("kurtosis")

        if skew is not None:
            skew_label = _interpret_skewness(skew)
            c1.metric(f"Skewness — _{skew_label}_", _fmt_num(skew))

        if kurt is not None:
            kurt_label = _interpret_kurtosis(kurt)
            c2.metric(f"Kurtosis — _{kurt_label}_", _fmt_num(kurt))

        # Inline distribution chart
        st.markdown("##### 📈 Distribution")
        col_chart, col_box = st.columns(2)
        with col_chart:
            fig = px.histogram(
                df, x=selected_col, nbins=40,
                title=f"Distribution of {selected_col}",
                color_discrete_sequence=["#6366F1"],
            )
            fig.update_layout(height=300, margin=dict(t=30, b=20))
            st.plotly_chart(fig, width='stretch')

        with col_box:
            fig = px.box(
                df, y=selected_col,
                title=f"Box Plot: {selected_col}",
                color_discrete_sequence=["#6366F1"],
            )
            fig.update_layout(height=300, margin=dict(t=30, b=20))
            st.plotly_chart(fig, width='stretch')

    # ══════════════════════════════════════════════════════
    #  Row 2b: Text Statistics (if applicable)
    # ══════════════════════════════════════════════════════

    if is_text:
        st.markdown("##### 📝 Text Statistics")

        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Length", _fmt_num(stats.get("avg_length")))
        c2.metric("Min Length", _fmt_num(stats.get("min_length")))
        c3.metric("Max Length", _fmt_num(stats.get("max_length")))

        c1, c2 = st.columns(2)
        ws = stats.get("has_whitespace_issues")
        mc = stats.get("has_mixed_case")

        if ws is not None:
            icon = "⚠️ Yes" if ws else "✅ No"
            c1.metric("Whitespace Issues", icon)
        if mc is not None:
            icon = "⚠️ Yes" if mc else "✅ No"
            c2.metric("Mixed Case", icon)

    # ══════════════════════════════════════════════════════
    #  Row 2c: Datetime Statistics (if applicable)
    # ══════════════════════════════════════════════════════

    if is_datetime:
        st.markdown("##### 📅 Datetime Statistics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Earliest", str(stats.get("min", "N/A")))
        c2.metric("Latest", str(stats.get("max", "N/A")))

        date_range = stats.get("date_range")
        if date_range:
            c3.metric("Range", str(date_range))

    # ══════════════════════════════════════════════════════
    #  Row 3: Most Common Values
    # ══════════════════════════════════════════════════════

    most_common = stats.get("most_common")
    if most_common:
        st.markdown("##### 🏆 Most Common Values")

        if isinstance(most_common, dict):
            mc_df = pd.DataFrame([
                {"Value": str(k), "Count": v}
                for k, v in most_common.items()
            ])
        elif isinstance(most_common, list):
            mc_df = pd.DataFrame([
                {"Value": str(item.get("value", item)),
                 "Count": item.get("count", "")}
                for item in most_common
            ]) if most_common and isinstance(most_common[0], dict) else pd.DataFrame(
                {"Value": [str(v) for v in most_common]}
            )
        else:
            mc_df = None

        if mc_df is not None and not mc_df.empty:
            # Horizontal bar chart for top values
            if "Count" in mc_df.columns and len(mc_df) > 1:
                fig = px.bar(
                    mc_df.head(15),
                    x="Count", y="Value",
                    orientation="h",
                    title="Top Values",
                    color_discrete_sequence=["#8B5CF6"],
                )
                fig.update_layout(
                    height=max(200, len(mc_df.head(15)) * 28 + 60),
                    margin=dict(t=30, b=20),
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig, width='stretch')
            else:
                st.dataframe(mc_df, width='stretch', hide_index=True)

    # ══════════════════════════════════════════════════════
    #  Row 4: Sample Values
    # ══════════════════════════════════════════════════════

    st.markdown("##### 📄 Sample Values")
    sample_df = df[[selected_col]].head(20)
    st.dataframe(sample_df, width='stretch', hide_index=True)


# ══════════════════════════════════════════════════════════════
#  Formatting Helpers
# ══════════════════════════════════════════════════════════════

def _fmt_num(val) -> str:
    """Format a numeric value for display."""
    if val is None:
        return "N/A"
    if isinstance(val, float):
        if abs(val) >= 1_000_000:
            return f"{val:,.0f}"
        elif abs(val) >= 100:
            return f"{val:,.2f}"
        elif abs(val) >= 1:
            return f"{val:.3f}"
        else:
            return f"{val:.4f}"
    return str(val)


def _interpret_skewness(skew: float) -> str:
    """Return human-readable skewness interpretation."""
    if abs(skew) < 0.5:
        return "Symmetric"
    elif skew > 0:
        return "Right-skewed" if skew < 1 else "Heavily right-skewed"
    else:
        return "Left-skewed" if skew > -1 else "Heavily left-skewed"


def _interpret_kurtosis(kurt: float) -> str:
    """Return human-readable kurtosis interpretation."""
    if abs(kurt) < 0.5:
        return "Normal tails"
    elif kurt > 0:
        return "Heavy tails" if kurt < 3 else "Very heavy tails"
    else:
        return "Light tails"


def _dtype_color(dtype: str) -> str:
    """Return a color for a data type badge."""
    if "int" in dtype:
        return "#3B82F6"
    elif "float" in dtype:
        return "#6366F1"
    elif "datetime" in dtype:
        return "#F59E0B"
    elif "bool" in dtype:
        return "#10B981"
    elif "category" in dtype:
        return "#EC4899"
    elif dtype in ("object", "string"):
        return "#8B5CF6"
    else:
        return "#6B7280"


main()
