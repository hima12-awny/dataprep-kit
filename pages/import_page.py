"""
Page 1: Data Import
Upload CSV, Excel, JSON, or Parquet files with configurable options.
"""

import streamlit as st

from core.state import StateManager
from core.dataset import Dataset
from core.pipeline import Pipeline
from data_io.importers import DataImporter
from data_io.pipeline_io import PipelineIO
from config.settings import settings



def main():

    st.markdown(
        """
        <div class="page-header">
            <h1>📂 Import Data</h1>
            <p>Upload your dataset or import a saved pipeline</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2 = st.tabs(["📤 Upload Dataset", "📋 Import Pipeline"])

    with tab1:
        _render_upload_tab()

    with tab2:
        _render_pipeline_import_tab()


def _render_upload_tab():
    """Render the file upload interface."""
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=list(settings.SUPPORTED_FILE_TYPES),
        help=f"Supported formats: {', '.join(settings.SUPPORTED_FILE_TYPES)}",
    )

    if uploaded_file is None:
        st.info("👆 Upload a file to get started.")
        return

    file_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name
    ext = filename.rsplit(".", 1)[-1].lower()

    st.success(f"📄 **{filename}** ({len(file_bytes) / 1024:.1f} KB)")

    # ── Format-specific options ───────────────────────────────
    with st.expander("⚙️ Import Options", expanded=True):
        if ext == "csv":
            col1, col2, col3 = st.columns(3)
            with col1:
                delimiter = st.selectbox(
                    "Delimiter",
                    options=settings.SUPPORTED_DELIMITERS,
                    format_func=lambda d: {
                        ",": "Comma (,)", ";": "Semicolon (;)", "\t": "Tab", "|": "Pipe (|)"}.get(d, d),
                )
            with col2:
                encoding = st.selectbox(
                    "Encoding", options=settings.SUPPORTED_ENCODINGS)
            with col3:
                header_row = st.number_input(
                    "Header row", min_value=0, value=0, help="Row number for column names")

            col1, col2 = st.columns(2)
            with col1:
                sample_enabled = st.checkbox("Sample rows", value=False)
            with col2:
                sample_rows = st.number_input(
                    "Number of rows",
                    min_value=100, value=settings.SAMPLE_ROWS_DEFAULT,
                    disabled=not sample_enabled,
                )

        elif ext in ("xlsx", "xls"):
            try:
                sheet_names = DataImporter.get_excel_sheet_names(file_bytes)
                sheet_name = st.selectbox("Select sheet", options=sheet_names)
            except Exception:
                sheet_name = 0
                st.warning("Could not read sheet names. Using first sheet.")

            header_row = st.number_input("Header row", min_value=0, value=0)
            sample_enabled = st.checkbox("Sample rows", value=False)
            sample_rows = st.number_input(
                "Number of rows", min_value=100, value=settings.SAMPLE_ROWS_DEFAULT,
                disabled=not sample_enabled,
            )

        elif ext == "json":
            col1, col2 = st.columns(2)
            with col1:
                orient = st.selectbox(
                    "JSON orient", ["records", "columns", "index", "split"])
            with col2:
                lines = st.checkbox("Line-delimited JSON", value=False)

    # ── Import Button ─────────────────────────────────────────
    if st.button("🚀 Import Data", type="primary", width='stretch'):
        with st.spinner("Importing..."):
            try:
                importer = DataImporter

                if ext == "csv":
                    df, config = importer.import_csv(
                        file_bytes,
                        delimiter=delimiter,
                        encoding=encoding,
                        header_row=int(header_row),
                        sample_rows=int(
                            sample_rows) if sample_enabled else None,
                    )
                elif ext in ("xlsx", "xls"):
                    df, config = importer.import_excel(
                        file_bytes,
                        sheet_name=sheet_name,
                        header_row=int(header_row),
                        sample_rows=int(
                            sample_rows) if sample_enabled else None,
                    )
                elif ext == "json":
                    df, config = importer.import_json(
                        file_bytes, orient=orient, lines=lines)
                elif ext == "parquet":
                    df, config = importer.import_parquet(file_bytes)
                else:
                    st.error(f"Unsupported format: .{ext}")
                    return

                # Create dataset and store in state
                dataset = Dataset(df=df, source_name=filename,
                                  import_config=config)
                StateManager.set_dataset(dataset)
                StateManager.set_pipeline(Pipeline())

                st.success(
                    f"✅ Imported **{filename}** — {len(df):,} rows × {len(df.columns)} columns")
                st.dataframe(df.head(settings.PREVIEW_ROWS),
                             width='stretch')

            except Exception as e:
                st.error(f"❌ Import failed: {e}")


def _render_pipeline_import_tab():
    """Render the pipeline JSON import interface."""
    st.markdown(
        """
        Import a previously exported pipeline JSON file to replay
        the same transformations on a new dataset.
        """
    )

    pipeline_file = st.file_uploader(
        "Upload pipeline JSON",
        type=["json"],
        key="pipeline_upload",
    )

    if pipeline_file is None:
        return

    file_bytes = pipeline_file.getvalue()

    try:
        pipeline, errors = PipelineIO.import_pipeline_from_bytes(file_bytes)

        if errors:
            for err in errors:
                st.error(f"⚠️ {err}")
            return

        if pipeline is None:
            st.error("Failed to parse pipeline.")
            return

        st.success(
            f"✅ Pipeline loaded: **{pipeline.pipeline_id}** — {pipeline.step_count} steps")

        # Preview steps
        for i, step in enumerate(pipeline.steps):
            status = "✅" if step.enabled else "⏸️"
            st.markdown(
                f"  {status} **Step {i+1}:** {step.description} (`{step.action_type}`)")

        # Apply pipeline
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📋 Load Pipeline Only", width='stretch'):
                StateManager.set_pipeline(pipeline)
                StateManager.add_notification(
                    f"Pipeline loaded with {pipeline.step_count} steps", "success"
                )
                st.rerun()

        with col2:
            has_dataset = StateManager.has_dataset()
            if st.button(
                "▶️ Load & Execute on Current Data",
                width='stretch',
                disabled=not has_dataset,
            ):
                StateManager.set_pipeline(pipeline)
                dataset = StateManager.get_dataset()
                report = pipeline.execute(dataset)
                StateManager.invalidate_profiling_cache()

                if report["failed"] > 0:
                    st.warning(
                        f"Executed {report['executed']}/{report['total_steps']} steps. "
                        f"{report['failed']} failed."
                    )
                    for err in report.get("errors", []):
                        st.error(f"  → {err['error']}")
                else:
                    StateManager.add_notification(
                        f"Pipeline executed: {report['executed']} steps successful", "success"
                    )
                    st.rerun()

            if not has_dataset:
                st.caption(
                    "⬆️ Import a dataset first to execute the pipeline.")

    except Exception as e:
        st.error(f"❌ Failed to import pipeline: {e}")


main()
