"""
Page 7: Export
Download cleaned data and pipeline JSON.
"""

import streamlit as st
import json

from core.state import StateManager
from data_io.exporters import DataExporter
from data_io.pipeline_io import PipelineIO
from components.metrics_bar import render_metrics_bar
from components.code_exporter import render_code_export
from config.settings import settings


def main():

    st.markdown(
        """
        <div class="page-header">
            <h1>💾 Export</h1>
            <p>Download your cleaned data and reproducible pipeline</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not StateManager.has_dataset():
        st.warning("⚠️ No dataset loaded. Go to the **Import** page first.")
        return

    dataset = StateManager.get_dataset()
    pipeline = StateManager.get_pipeline()
    render_metrics_bar()
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs([
        "📊 Export Data",
        "📋 Export Pipeline",
        "🐍 Export Code",
    ])

    with tab1:
        _render_data_export(dataset)
    with tab2:
        _render_pipeline_export(pipeline, dataset)
    with tab3:
        _render_code_export(pipeline)


def _render_data_export(dataset):
    """Data export interface."""
    st.markdown("### 📊 Export Cleaned Data")

    df = dataset.df
    overview = dataset.overview()

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{overview['row_count']:,}")
    col2.metric("Columns", f"{overview['column_count']}")
    col3.metric("Missing %", f"{overview['missing_percentage']:.1f}%")

    st.markdown("---")

    # Format selection
    col1, col2 = st.columns(2)
    with col1:
        export_format = st.selectbox(
            "Export format",
            options=list(settings.EXPORT_FORMATS),
            format_func=lambda f: {
                "csv": "CSV (.csv)",
                "xlsx": "Excel (.xlsx)",
                "json": "JSON (.json)",
                "parquet": "Parquet (.parquet)",
            }.get(f, f),
            key="export_format",
        )

    with col2:
        filename_base = dataset.source_name.rsplit(
            ".", 1)[0] if "." in dataset.source_name else dataset.source_name
        filename = st.text_input(
            "Filename",
            value=f"{filename_base}_cleaned",
            key="export_filename",
        )

    # Format-specific options
    include_index = False
    csv_delimiter = ","
    json_orient = "records"

    if export_format == "csv":
        col1, col2 = st.columns(2)
        with col1:
            csv_delimiter = st.selectbox(
                "Delimiter", [",", ";", "\t", "|"], key="exp_delim")
        with col2:
            include_index = st.checkbox(
                "Include index", value=False, key="exp_idx")

    elif export_format == "json":
        json_orient = st.selectbox(
            "JSON orient", ["records", "columns", "index", "split"], key="exp_orient")

    elif export_format == "xlsx":
        include_index = st.checkbox(
            "Include index", value=False, key="exp_idx_xl")

    # Preview
    with st.expander("👁️ Preview (first 20 rows)"):
        st.dataframe(df.head(20), width='stretch')

    # Download button
    st.markdown("---")

    try:
        if export_format == "csv":
            file_bytes = DataExporter.to_csv(
                df, index=include_index, delimiter=csv_delimiter)
        elif export_format == "xlsx":
            file_bytes = DataExporter.to_excel(df, index=include_index)
        elif export_format == "json":
            file_bytes = DataExporter.to_json(df, orient=json_orient)
        elif export_format == "parquet":
            file_bytes = DataExporter.to_parquet(df, index=include_index)
        else:
            st.error(f"Unsupported format: {export_format}")
            return

        ext = DataExporter.get_file_extension(export_format)
        mime = DataExporter.get_mime_type(export_format)

        st.download_button(
            label=f"📥 Download {filename}.{ext}",
            data=file_bytes,
            file_name=f"{filename}.{ext}",
            mime=mime,
            width='stretch',
            type="primary",
        )

        st.caption(f"File size: ~{len(file_bytes) / 1024:.1f} KB")

    except Exception as e:
        st.error(f"❌ Export failed: {e}")


def _render_pipeline_export(pipeline, dataset):
    """Pipeline JSON export interface."""
    st.markdown("### 📋 Export Pipeline")
    st.markdown(
        """
        Export your entire data preparation pipeline as a JSON file.
        This file can be imported later to **replay the same transformations**
        on new data.
        """
    )

    if pipeline.step_count == 0:
        st.info(
            "📋 Pipeline is empty. Add actions from the Cleaning, Conversion, or Feature Engineering pages.")
        return

    # Add dataset metadata to pipeline
    pipeline.metadata = dataset.get_metadata()

    # Stats
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Steps", pipeline.step_count)
    col2.metric("Enabled Steps", len(pipeline.enabled_steps))

    # Count by author
    authors = {}
    for step in pipeline.steps:
        authors[step.author] = authors.get(step.author, 0) + 1
    author_str = ", ".join(f"{k}: {v}" for k, v in authors.items())
    col3.metric("Authors", author_str)

    st.markdown("---")

    # Step summary
    st.markdown("**Pipeline Steps:**")
    for i, step in enumerate(pipeline.steps):
        status = "✅" if step.enabled else "⏸️"
        st.markdown(
            f"  {i+1}. {status} **{step.description}** (`{step.action_type}`, author: `{step.author}`)")

    st.markdown("---")

    # Pipeline JSON preview
    with st.expander("🔍 JSON Preview"):
        st.json(pipeline.to_dict())

    # Download
    pipeline_json = PipelineIO.export_pipeline_bytes(pipeline)

    filename_base = dataset.source_name.rsplit(
        ".", 1)[0] if "." in dataset.source_name else "pipeline"

    st.download_button(
        label="📥 Download Pipeline JSON",
        data=pipeline_json,
        file_name=f"{filename_base}_pipeline.json",
        mime="application/json",
        width='stretch',
        type="primary",
    )

    st.caption(f"Pipeline ID: `{pipeline.pipeline_id}`")
    st.caption(f"Schema version: `{settings.PIPELINE_SCHEMA_VERSION}`")


def _render_code_export(pipeline):
    """Python code export."""
    st.markdown("### 🐍 Export as Python Script")
    st.markdown(
        "_Generate a standalone Python script that reproduces all pipeline steps._")

    render_code_export(pipeline)


main()
