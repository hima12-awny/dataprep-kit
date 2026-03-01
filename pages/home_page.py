"""Home page."""

import streamlit as st
import pandas as pd

from core.state import StateManager


st.markdown("## 🧪 DataPrep Kit")
st.caption("AI-powered data preparation toolkit for analytics & ML engineers")

st.markdown("---")

if not StateManager.has_dataset():
    st.markdown(
        """
        ### 👋 Welcome!

        DataPrep Kit helps you **clean, transform, and engineer features** from your data
        through an interactive, step-by-step pipeline.

        **Get started:**

        | Step | Page | Description |
        |------|------|-------------|
        | 1 | **Import Data** | Upload CSV, Excel, JSON, or Parquet files |
        | 2 | **Profiling** | Understand your data's structure and quality |
        | 3 | **Cleaning** | Fix missing values, duplicates, outliers, text |
        | 4 | **Conversion** | Cast types, parse dates, encode categories |
        | 5 | **Feature Engineering** | Create new features from existing columns |
        | 6 | **Pipeline** | Review, reorder, and manage your steps |
        | 7 | **Export** | Download data + pipeline JSON for replay |

        **Optional:** Configure an **AI Agent** in **AI Settings** to get smart,
        context-aware recommendations powered by LLMs.
        """
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 🤖 Smart Recommendations")
        st.markdown(
            "Static rules detect issues automatically. AI Agent gives deeper, context-aware advice.")
    with c2:
        st.markdown("#### 🔄 Reproducible Pipelines")
        st.markdown(
            "Export your entire workflow as JSON. Import and replay on new datasets.")
    with c3:
        st.markdown("#### ↩️ Full Undo / Redo")
        st.markdown(
            "Every action is reversible. Preview changes before applying.")

else:
    dataset = StateManager.get_dataset()
    overview = dataset.overview()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🔢 Rows", f"{overview['row_count']:,}")
    c2.metric("📊 Columns", f"{overview['column_count']}")
    c3.metric("🕳️ Missing", f"{overview['missing_percentage']:.1f}%")
    c4.metric("👥 Duplicates", f"{overview['duplicate_rows']:,}")
    c5.metric("💾 Memory", f"{overview['memory_usage_mb']:.1f} MB")

    st.markdown("---")

    st.markdown("### Quick Preview")
    st.dataframe(dataset.head(20), width='content', height=350)

    st.markdown("### Column Summary")
    col_info = []
    for col_name in dataset.columns:
        stats = dataset.column_stats(col_name)
        col_info.append({
            "Column": col_name,
            "Type": stats.get("dtype", "?"),
            "Non-Null": stats.get("non_null_count", 0),
            "Null %": f"{stats.get('null_percentage', 0):.1f}%",
            "Unique": stats.get("unique_count", 0),
        })
    st.dataframe(pd.DataFrame(col_info), width='content', hide_index=True)
