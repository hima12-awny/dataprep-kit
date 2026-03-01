"""
Pipeline step list viewer with controls for reorder, delete, and toggle.
"""

import streamlit as st
from typing import Optional, Callable

from core.pipeline import Pipeline, PipelineStep


def render_pipeline_viewer(
    pipeline: Pipeline,
    on_delete: Optional[Callable] = None,
    on_toggle: Optional[Callable] = None,
    on_reorder: Optional[Callable] = None,
):
    """
    Render the pipeline as an interactive list of steps.
    """
    if pipeline.step_count == 0:
        st.info(
            "📋 Pipeline is empty. Add actions from the Cleaning, "
            "Conversion, or Feature Engineering pages."
        )
        return

    st.markdown(f"### Pipeline Steps ({pipeline.step_count})")

    for i, step in enumerate(pipeline.steps):
        _render_step_row(
            step=step,
            index=i,
            total_steps=pipeline.step_count,
            on_delete=on_delete,
            on_toggle=on_toggle,
            on_reorder=on_reorder,
        )


def _render_step_row(
    step: PipelineStep,
    index: int,
    total_steps: int,
    on_delete: Optional[Callable],
    on_toggle: Optional[Callable],
    on_reorder: Optional[Callable],
):
    """Render a single pipeline step as a clean card."""

    status_icon = "✅" if step.enabled else "⏸️"
    author_label = _get_author_label(step.author)
    action_icon = _get_action_icon(step.action_type)

    # ── Step Card Container ───────────────────────────────────
    with st.container():
        # Row 1: Step info
        st.markdown(
            f"""
            <div style="
                border: 1px solid #374151;
                border-radius: 10px;
                padding: 0.85rem 1.1rem;
                margin-bottom: 0.4rem;
                opacity: {'1' if step.enabled else '0.55'};
                border-left: 4px solid {'#10B981' if step.enabled else '#6B7280'};
            ">
                <div style="display: flex; align-items: flex-start; gap: 0.6rem;">
                    <span style="
                        font-weight: 700;
                        font-size: 0.9rem;
                        color: #9CA3AF;
                        min-width: 28px;
                    ">{index + 1}.</span>
                    <span style="font-size: 1rem;">{status_icon} {action_icon}</span>
                    <div style="flex: 1; min-width: 0;">
                        <div style="
                            font-weight: 600;
                            font-size: 0.92rem;
                            word-wrap: break-word;
                            overflow-wrap: break-word;
                            line-height: 1.4;
                        ">{step.description}</div>
                        <div style="
                            font-size: 0.78rem;
                            color: #9CA3AF;
                            margin-top: 0.2rem;
                        ">
                            <code>{step.action_type}</code> · {author_label}
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Row 2: Action buttons
        col_up, col_down, col_toggle, col_delete = st.columns(4)

        with col_up:
            st.button(
                "Move Up",
                key=f"up_{step.action_id}",
                disabled=index == 0,
                width='stretch',
                on_click=_handle_reorder,
                args=(on_reorder, step.action_id, index - 1),
            )

        with col_down:
            st.button(
                "Move Down",
                key=f"down_{step.action_id}",
                disabled=index >= total_steps - 1,
                width='stretch',
                on_click=_handle_reorder,
                args=(on_reorder, step.action_id, index + 1),
            )

        with col_toggle:
            toggle_label = "Disable" if step.enabled else "Enable"
            st.button(
                toggle_label,
                key=f"toggle_{step.action_id}",
                width='stretch',
                on_click=_handle_toggle,
                args=(on_toggle, step.action_id),
            )

        with col_delete:
            st.button(
                "Delete",
                key=f"delete_{step.action_id}",
                width='stretch',
                on_click=_handle_delete,
                args=(on_delete, step.action_id),
            )

        # Row 3: Expandable details
        with st.expander("📝 Details"):
            detail_col1, detail_col2 = st.columns(2)
            with detail_col1:
                st.markdown(f"**ID:** `{step.action_id}`")
                st.markdown(f"**Type:** `{step.action_type}`")
            with detail_col2:
                st.markdown(f"**Author:** `{step.author}`")
                st.markdown(f"**Time:** `{step.timestamp}`")

            st.markdown("**Parameters:**")
            st.json(step.parameters)

        st.markdown("")  # Spacer between steps


# ── Callback Handlers ─────────────────────────────────────────
# Using on_click callbacks avoids the "stale state" issue with
# buttons inside loops.

def _handle_reorder(callback, action_id, new_index):
    if callback:
        callback(action_id, new_index)


def _handle_toggle(callback, action_id):
    if callback:
        callback(action_id)


def _handle_delete(callback, action_id):
    if callback:
        callback(action_id)


# ── Helpers ───────────────────────────────────────────────────

def _get_author_label(author: str) -> str:
    """Return a styled HTML label for the author."""
    colors = {
        "user": "#3B82F6",
        "ai_static": "#8B5CF6",
        "ai_agent": "#EC4899",
        "ai": "#8B5CF6",
    }
    labels = {
        "user": "User",
        "ai_static": "AI Static",
        "ai_agent": "AI Agent",
        "ai": "AI",
    }
    color = colors.get(author, "#6B7280")
    label = labels.get(author, author)
    return (
        f'<span style="'
        f"background-color: {color};"
        f"color: white;"
        f"padding: 1px 8px;"
        f"border-radius: 10px;"
        f"font-size: 0.72rem;"
        f"font-weight: 600;"
        f"letter-spacing: 0.02em;"
        f'">{label}</span>'
    )


def _get_action_icon(action_type: str) -> str:
    """Return emoji icon for action type."""
    icons = {
        "handle_missing": "🕳️",
        "handle_duplicates": "👥",
        "handle_outliers": "📊",
        "text_cleaning": "✏️",
        "inconsistency": "🔧",
        "type_casting": "🔄",
        "datetime_ops": "📅",
        "numeric_transform": "🔢",
        "encoding": "🏷️",
        "column_ops": "📐",
        "aggregation": "📈",
        "temporal": "⏱️",
        "interaction": "🔗",
    }
    return icons.get(action_type, "⚙️")
