"""
Page 6: Pipeline Management
View all steps, reorder, undo/redo, change log, import/export pipeline.
"""

import streamlit as st

from core.state import StateManager
from core.pipeline import Pipeline
from components.pipeline_viewer import render_pipeline_viewer
from components.code_exporter import render_code_export
from data_io.pipeline_io import PipelineIO


def main():


    st.markdown(
        """
        <div class="page-header">
            <h1>📋 Pipeline Manager</h1>
            <p>Review, reorder, and manage your data preparation pipeline</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pipeline = StateManager.get_pipeline()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Pipeline Steps",
        "🐍 Python Code",
        "📜 Change Log",
        "🔄 Re-Execute",
    ])

    with tab1:
        _render_pipeline_steps(pipeline)
    with tab2:
        _render_code_tab(pipeline)
    with tab3:
        _render_change_log(pipeline)
    with tab4:
        _render_reexecute(pipeline)


def _render_pipeline_steps(pipeline: Pipeline):
    """Render the interactive pipeline step list."""
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.markdown(
            f"**{pipeline.step_count} steps** ({len(pipeline.enabled_steps)} enabled)"
        )

    with col2:
        if st.button(
            "Clear All",
            width='stretch',
            disabled=pipeline.step_count == 0,
        ):
            pipeline.clear()
            StateManager.add_notification("Pipeline cleared", "warning")
            st.rerun()

    with col3:
        if st.button(
            "Duplicate Pipeline",
            width='stretch',
            disabled=pipeline.step_count == 0,
        ):
            new_pipeline = pipeline.duplicate()
            StateManager.set_pipeline(new_pipeline)
            StateManager.add_notification(
                "Pipeline duplicated with new IDs", "info")
            st.rerun()

    st.markdown("---")

    render_pipeline_viewer(
        pipeline=pipeline,
        on_delete=lambda aid: _handle_delete(pipeline, aid),
        on_toggle=lambda aid: _handle_toggle(pipeline, aid),
        on_reorder=lambda aid, idx: _handle_reorder(pipeline, aid, idx),
    )

    # Pipeline JSON preview
    if pipeline.step_count > 0:
        with st.expander("🔍 Pipeline JSON Preview"):
            st.json(pipeline.to_dict())


def _render_code_tab(pipeline: Pipeline):
    """Render reproducible Python code."""
    st.markdown("### 🐍 Reproducible Python Code")
    st.markdown(
        "_This code reproduces all enabled pipeline steps using pandas._")
    render_code_export(pipeline)


def _render_change_log(pipeline: Pipeline):
    """Render the pipeline audit trail."""
    st.markdown("### 📜 Change Log")

    log = pipeline.get_change_log()
    if not log:
        st.info("No changes recorded yet.")
        return

    for entry in reversed(log):
        icon_map = {
            "add": "➕",
            "remove": "🗑️",
            "toggle": "🔀",
            "reorder": "↕️",
            "clear": "🧹",
        }
        icon = icon_map.get(entry.get("change_type", ""), "📝")
        st.markdown(
            f"**{icon} {entry.get('change_type', '').title()}** — "
            f"{entry.get('description', '')}  \n"
            f"_`{entry.get('timestamp', '')}`_ · "
            f"Action ID: `{entry.get('action_id', '')}`"
        )
        st.markdown("---")


def _render_reexecute(pipeline: Pipeline):
    """Re-execute the entire pipeline on the current dataset."""
    st.markdown("### 🔄 Re-Execute Pipeline")
    st.markdown(
        """
        Re-run all enabled pipeline steps from scratch on the
        **original imported data**. This is useful after reordering
        or toggling steps.
        """
    )

    dataset = StateManager.get_dataset()
    if not dataset:
        st.warning("No dataset loaded.")
        return

    if pipeline.step_count == 0:
        st.info("No pipeline steps to execute.")
        return

    st.markdown(
        f"**{len(pipeline.enabled_steps)}** enabled steps will be executed.")

    for i, step in enumerate(pipeline.enabled_steps):
        status = "✅" if step.enabled else "⏸️"
        st.markdown(
            f"  {i+1}. {status} {step.description} (`{step.action_type}`)"
        )

    st.markdown("---")

    if st.button(
        "▶️ Re-Execute All Enabled Steps",
        type="primary",
        width='stretch',
    ):
        with st.spinner("Executing pipeline..."):
            # Restore to original state
            if dataset._undo_stack:
                original_df = dataset._undo_stack[0].dataframe.copy()
                dataset._df = original_df
                dataset._undo_stack = dataset._undo_stack[:1]
                dataset._redo_stack.clear()

            report = pipeline.execute(dataset)
            StateManager.invalidate_profiling_cache()

            if report["failed"] > 0:
                st.warning(
                    f"Executed {report['executed']}/{report['total_steps']} steps. "
                    f"{report['failed']} failed."
                )
                for err in report.get("errors", []):
                    st.error(
                        f"  → Step `{err.get('action_id')}`: {err.get('error')}"
                    )
            else:
                StateManager.add_notification(
                    f"Pipeline re-executed: {report['executed']} steps successful",
                    "success",
                )
                st.rerun()


# ── Handlers ──────────────────────────────────────────────────

def _handle_delete(pipeline: Pipeline, action_id: str):
    pipeline.remove_step(action_id)
    StateManager.add_notification("Step removed", "warning")


def _handle_toggle(pipeline: Pipeline, action_id: str):
    pipeline.toggle_step(action_id)


def _handle_reorder(pipeline: Pipeline, action_id: str, new_index: int):
    pipeline.reorder_step(action_id, new_index)


main()
