"""
Page 4: Data Conversion
Type casting, datetime operations, numeric transforms, and encoding.
Shows static + AI recommendations with approve/preview/reject controls.
"""

import streamlit as st
import pandas as pd

from core.state import StateManager
from core.pipeline import PipelineStep
from recommendations.engine import RecommendationEngine
from components.metrics_bar import render_metrics_bar
from components.action_list import render_action_list
from components.preview_table import render_preview_table
from components.column_selector import render_column_selector
from config.registry import ActionRegistry


def main():

    st.markdown(
        """
        <div class="page-header">
            <h1>🔄 Data Conversion</h1>
            <p>Convert data types, parse dates, transform numbers, and encode categories</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not StateManager.has_dataset():
        st.warning("⚠️ No dataset loaded. Go to the **Import** page first.")
        return

    dataset = StateManager.get_dataset()
    render_metrics_bar()
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🤖 Recommendations",
        "🔄 Type Casting",
        "📅 Datetime",
        "🔢 Numeric",
        "🏷️ Encoding",
    ])

    with tab1:
        _render_recommendations(dataset)
    with tab2:
        _render_type_casting(dataset)
    with tab3:
        _render_datetime(dataset)
    with tab4:
        _render_numeric(dataset)
    with tab5:
        _render_encoding(dataset)


# ══════════════════════════════════════════════════════════════
#  Helper: Action Signature for Deduplication
# ══════════════════════════════════════════════════════════════

def _action_signature(rec: dict) -> str:
    """
    Create a deduplication signature from a recommendation.
    Used to track which recommendations have already been approved,
    so the static analyzer doesn't re-recommend them after refresh.
    """
    action_type = rec.get("action_type", "")
    params = rec.get("parameters", {})

    columns = (
        params.get("columns")
        or params.get("subset")
        or ([params["column"]] if params.get("column") else [])
    )
    if isinstance(columns, str):
        columns = [columns]

    # Also handle conversions dict (type_casting uses {col: type})
    if not columns and params.get("conversions"):
        columns = list(params["conversions"].keys())

    columns = sorted(columns or [])

    strategy = (
        params.get("strategy")
        or params.get("method")
        or params.get("operation")
        or ""
    )

    # Include target type for type casting
    target = ""
    if params.get("conversions"):
        target = "|".join(sorted(params["conversions"].values()))

    return f"{action_type}|{'|'.join(columns)}|{strategy}|{target}"


# ══════════════════════════════════════════════════════════════
#  Tab 1: Recommendations (Static + AI)
# ══════════════════════════════════════════════════════════════

def _render_recommendations(dataset):
    """Show conversion recommendations from static analysis and optional AI agent."""
    st.markdown("### 🤖 Smart Recommendations")
    st.markdown(
        "_Type conversions detected via regex and heuristic analysis. "
        "Enable AI for deeper, context-aware suggestions._"
    )

    # ── Initialize trackers ───────────────────────────────
    if "conv_applied_actions" not in st.session_state:
        st.session_state["conv_applied_actions"] = set()

    # ── Inline success / error feedback ───────────────────
    last_applied = st.session_state.pop("conv_last_applied", None)
    if last_applied:
        st.success(f"✅ Successfully applied: **{last_applied}**")

    last_error = st.session_state.pop("conv_last_error", None)
    if last_error:
        st.error(f"❌ Action failed: {last_error}")

    # ── AI Configuration Status ───────────────────────────
    ai_configured = StateManager.is_ai_configured()
    ai_config = StateManager.get_ai_config()

    col_toggle, col_status = st.columns([1, 2])

    with col_toggle:
        use_ai = st.checkbox(
            "🤖 Include AI Recommendations",
            key="conv_use_ai",
            disabled=not ai_configured,
            help="Uses your configured AI model to suggest additional conversion actions",
        )

    with col_status:
        if ai_configured:
            provider = ai_config.get("provider", "")
            model = ai_config.get("model", "")
            st.success(f"✅ AI Ready: **{provider}** / **{model}**")
        else:
            st.info(
                "🔑 Set up your API key on the **AI Settings** page "
                "to enable AI-powered recommendations."
            )

    if use_ai and ai_configured and not StateManager.has_data_description():
        st.caption(
            "💡 **Tip:** Add a data description in **AI Settings** "
            "for more targeted AI recommendations."
        )

    # ── Generate / Refresh Button ─────────────────────────
    if st.button("🔄 Generate Recommendations", key="refresh_conv_recs"):
        st.session_state.pop("conv_recs_cache", None)
        st.session_state.pop("conv_rec_preview_data", None)
        st.session_state.pop("conv_recs_errors", None)
        st.session_state["conv_applied_actions"] = set()
        st.session_state["conv_force_generate"] = True
        st.rerun()

    # ── Generate / Cache Recommendations ──────────────────
    if "conv_recs_cache" not in st.session_state:
        engine = RecommendationEngine()

        force = st.session_state.pop("conv_force_generate", False)
        include_ai = use_ai and ai_configured and force
        effective_ai_config = ai_config if include_ai else None

        if effective_ai_config:
            with st.spinner(
                "🤖 AI is analyzing your dataset for conversion opportunities… "
                "This may take a moment."
            ):
                recs = engine.get_conversion_recommendations(
                    dataset, ai_config=effective_ai_config
                )
        else:
            recs = engine.get_conversion_recommendations(dataset)

        # Separate errors from actionable recommendations
        errors = [r for r in recs if r.get("action_type") == "error"]
        actionable = [r for r in recs if r.get("action_type") != "error"]

        # Filter out already-applied actions
        applied = st.session_state.get("conv_applied_actions", set())
        if applied:
            actionable = [
                r for r in actionable
                if _action_signature(r) not in applied
            ]

        st.session_state["conv_recs_cache"] = actionable
        st.session_state["conv_recs_ai_included"] = include_ai
        st.session_state["conv_recs_errors"] = errors

    recs = st.session_state.get("conv_recs_cache", [])
    ai_included = st.session_state.get("conv_recs_ai_included", False)
    errors = st.session_state.get("conv_recs_errors", [])

    # ── Display Errors ────────────────────────────────────
    if errors:
        for err in errors:
            author = err.get("author", "system")
            badge = "🤖 AI Agent" if author == "ai_agent" else "⚙️ Analyzer"
            st.warning(f"⚠️ {badge}: {err.get('description', 'Unknown error')}")

    # ── Informational Messages ────────────────────────────
    if use_ai and ai_configured and not ai_included:
        st.info(
            "ℹ️ AI recommendations are not loaded yet. "
            "Click **Generate Recommendations** to include AI analysis."
        )

    # ── Recommendation Count Summary ──────────────────────
    if recs:
        static_count = sum(
            1 for r in recs
            if r.get("author") in ("ai_static", "system", None)
        )
        ai_count = sum(1 for r in recs if r.get("author") == "ai_agent")
        both_count = sum(1 for r in recs if r.get("author") == "both")

        parts = []
        if static_count:
            parts.append(f"📊 **{static_count}** static")
        if ai_count:
            parts.append(f"🤖 **{ai_count}** AI")
        if both_count:
            parts.append(f"⚡ **{both_count}** confirmed by both engines")
        if parts:
            st.caption(" · ".join(parts))

    # ── Render Recommendation Cards ───────────────────────
    render_action_list(
        recommendations=recs,
        on_approve=lambda r: _approve_action(r, dataset),
        on_reject=lambda r: _reject_action(r),
        on_preview=lambda r: _preview_action(r, dataset),
        title="Conversion Recommendations",
        key_prefix="conv_rec",
    )


# ══════════════════════════════════════════════════════════════
#  Tab 2: Type Casting
# ══════════════════════════════════════════════════════════════

def _render_type_casting(dataset):
    """Type casting interface."""
    st.markdown("### 🔄 Type Casting")
    df = dataset.df

    st.markdown("**Current column types:**")
    type_info = pd.DataFrame({
        "Column": df.columns,
        "Current Type": [str(df[col].dtype) for col in df.columns],
        "Sample Value": [
            str(df[col].dropna().iloc[0]) if df[col].notna().any() else "N/A"
            for col in df.columns
        ],
    })
    st.dataframe(type_info, width='stretch', hide_index=True)

    st.markdown("---")

    with st.form("type_cast_form"):
        col = st.selectbox("Select column", df.columns.tolist(), key="tc_col")
        target_type = st.selectbox(
            "Target type",
            ["int64", "float64", "string", "bool", "datetime64[ns]", "category"],
            key="tc_type",
        )

        date_format = None
        if target_type == "datetime64[ns]":
            date_format = st.text_input(
                "Date format (optional)", placeholder="%Y-%m-%d", key="tc_datefmt")

        errors = st.selectbox(
            "Error handling", ["coerce", "raise", "ignore"], key="tc_errors")

        submitted = st.form_submit_button(
            "✅ Convert", width='stretch')

        if submitted:
            params = {
                "conversions": {col: target_type},
                "errors": errors,
            }
            if date_format:
                params["date_format"] = date_format

            _execute_user_action(
                "type_casting", params, dataset,
                f"Convert '{col}' to {target_type}"
            )


# ══════════════════════════════════════════════════════════════
#  Tab 3: Datetime
# ══════════════════════════════════════════════════════════════

def _render_datetime(dataset):
    """Datetime operations interface."""
    st.markdown("### 📅 Datetime Operations")
    df = dataset.df

    all_cols = df.columns.tolist()

    with st.form("datetime_form"):
        operation = st.selectbox(
            "Operation",
            ["extract_components", "date_diff",
                "to_unix_timestamp", "from_unix_timestamp"],
            key="dt_op",
        )

        column = st.selectbox("Source column", all_cols, key="dt_col")

        components = []
        column2 = None
        unit = "days"

        if operation == "extract_components":
            components = st.multiselect(
                "Components to extract",
                ["year", "month", "day", "weekday", "day_name", "quarter",
                 "week", "is_weekend", "is_month_start", "hour", "minute"],
                default=["year", "month", "day", "weekday"],
                key="dt_comps",
            )
        elif operation == "date_diff":
            column2 = st.selectbox("Second column", all_cols, key="dt_col2")
            unit = st.selectbox(
                "Unit", ["days", "hours", "seconds"], key="dt_unit")

        submitted = st.form_submit_button("✅ Apply", width='stretch')

        if submitted:
            params = {
                "operation": operation,
                "column": column,
            }
            if operation == "extract_components":
                params["components"] = components
            elif operation == "date_diff":
                params["column2"] = column2
                params["unit"] = unit

            _execute_user_action(
                "datetime_ops", params, dataset,
                f"Datetime {operation} on '{column}'"
            )


# ══════════════════════════════════════════════════════════════
#  Tab 4: Numeric
# ══════════════════════════════════════════════════════════════

def _render_numeric(dataset):
    """Numeric transformations interface."""
    st.markdown("### 🔢 Numeric Transforms")
    df = dataset.df

    with st.form("numeric_form"):
        columns = render_column_selector(
            df, key="num_cols", label="Select numeric columns",
            type_filter="numeric",
        )

        operation = st.selectbox(
            "Operation",
            ["log", "log1p", "sqrt", "square", "abs", "round",
             "normalize", "standardize", "robust_scale", "clip", "binning"],
            key="num_op",
        )

        overwrite = st.checkbox(
            "Overwrite original column", value=True, key="num_overwrite")

        decimals = None
        lower = upper = None
        n_bins = None

        if operation == "round":
            decimals = st.number_input(
                "Decimal places", min_value=0, value=2, key="num_dec")
        elif operation == "clip":
            col1, col2 = st.columns(2)
            with col1:
                lower = st.number_input(
                    "Lower bound", value=0.0, key="num_lower")
            with col2:
                upper = st.number_input(
                    "Upper bound", value=100.0, key="num_upper")
        elif operation == "binning":
            n_bins = st.number_input(
                "Number of bins", min_value=2, value=5, key="num_bins")

        submitted = st.form_submit_button("✅ Apply", width='stretch')

        if submitted and columns:
            params = {
                "columns": columns,
                "operation": operation,
                "overwrite": overwrite,
            }
            if decimals is not None:
                params["decimals"] = decimals
            if lower is not None:
                params["lower"] = lower
            if upper is not None:
                params["upper"] = upper
            if n_bins is not None:
                params["n_bins"] = n_bins

            _execute_user_action(
                "numeric_transform", params, dataset,
                f"Apply '{operation}' to {columns}"
            )


# ══════════════════════════════════════════════════════════════
#  Tab 5: Encoding
# ══════════════════════════════════════════════════════════════

def _render_encoding(dataset):
    """Categorical encoding interface."""
    st.markdown("### 🏷️ Categorical Encoding")
    df = dataset.df

    with st.form("encoding_form"):
        columns = render_column_selector(
            df, key="enc_cols", label="Select categorical columns",
            type_filter="categorical",
        )

        method = st.selectbox(
            "Encoding method",
            ["onehot", "label", "frequency", "ordinal", "binary", "target"],
            key="enc_method",
        )

        drop_first = False
        drop_original = True
        target_column = None

        if method == "onehot":
            drop_first = st.checkbox(
                "Drop first dummy", value=False, key="enc_dropfirst")
            drop_original = st.checkbox(
                "Drop original column", value=True, key="enc_droporig")
        elif method == "target":
            numeric_cols = df.select_dtypes(
                include=["number"]).columns.tolist()
            target_column = st.selectbox(
                "Target column", numeric_cols, key="enc_target")

        submitted = st.form_submit_button("✅ Apply", width='stretch')

        if submitted and columns:
            params = {
                "columns": columns,
                "method": method,
                "drop_first": drop_first,
                "drop_original": drop_original,
            }
            if target_column:
                params["target_column"] = target_column

            _execute_user_action(
                "encoding", params, dataset,
                f"Apply '{method}' encoding to {columns}"
            )


# ══════════════════════════════════════════════════════════════
#  Shared Action Handlers
# ══════════════════════════════════════════════════════════════

def _approve_action(recommendation: dict, dataset):
    """Approve and execute a recommended action.
    Removes ONLY this action from cache — preserves all other recs including AI.
    """
    step = PipelineStep.from_dict(recommendation)
    pipeline = StateManager.get_pipeline()
    pipeline.add_step(step)
    result = pipeline.execute_single_step(step, dataset)

    if result["status"] == "success":
        StateManager.invalidate_profiling_cache()

        # ── Track to prevent re-recommendation ────────────────
        sig = _action_signature(recommendation)
        applied = st.session_state.get("conv_applied_actions", set())
        applied.add(sig)
        st.session_state["conv_applied_actions"] = applied

        # ── Remove ONLY this action from cache ────────────────
        recs = st.session_state.get("conv_recs_cache", [])
        st.session_state["conv_recs_cache"] = [
            r for r in recs
            if r.get("action_id") != recommendation.get("action_id")
        ]

        # ── Store success feedback ────────────────────────────
        desc = recommendation.get("description", "Action")
        rows_before = result.get("rows_before", "?")
        rows_after = result.get("rows_after", "?")
        st.session_state["conv_last_applied"] = (
            f"{desc}  (rows: {rows_before} → {rows_after})"
        )

        st.session_state.pop("conv_rec_preview_data", None)
        st.session_state.pop("manual_conv_preview_data", None)

        StateManager.add_notification(
            f"✅ Applied: {desc}", "success"
        )
        st.rerun()
    else:
        error_msg = result.get("error", "Unknown error")
        st.session_state["conv_last_error"] = error_msg
        pipeline.remove_step(step.action_id)
        st.rerun()


def _reject_action(recommendation: dict):
    """Dismiss a recommendation.
    Removes ONLY this action from cache — preserves all others.
    """
    recs = st.session_state.get("conv_recs_cache", [])
    st.session_state["conv_recs_cache"] = [
        r for r in recs
        if r.get("action_id") != recommendation.get("action_id")
    ]
    st.session_state.pop("conv_rec_preview_data", None)
    st.rerun()


def _preview_action(recommendation: dict, dataset):
    """Preview a recommended action — stores keyed by action_id for inline rendering."""
    action_id = recommendation.get("action_id", "unknown")
    step = PipelineStep.from_dict(recommendation)
    pipeline = StateManager.get_pipeline()
    preview = pipeline.preview_step(step, dataset)
    st.session_state[f"conv_rec_preview_{action_id}"] = preview


def _execute_user_action(action_type: str, params: dict, dataset, description: str):
    """Execute a manually configured action from the manual tabs.
    Marks recs as stale but does NOT delete the cache.
    """
    pipeline = StateManager.get_pipeline()
    step = pipeline.add_action(
        action_type=action_type,
        description=description,
        parameters=params,
        author="user",
    )
    result = pipeline.execute_single_step(step, dataset)

    if result["status"] == "success":
        StateManager.invalidate_profiling_cache()

        # Track so it won't be re-recommended on next Generate
        sig = _action_signature({
            "action_type": action_type,
            "parameters": params,
        })
        applied = st.session_state.get("conv_applied_actions", set())
        applied.add(sig)
        st.session_state["conv_applied_actions"] = applied

        # Mark stale but do NOT delete cache
        st.session_state["conv_recs_stale"] = True

        st.session_state.pop("conv_rec_preview_data", None)
        st.session_state.pop("manual_conv_preview_data", None)

        st.session_state["conv_last_applied"] = description
        StateManager.add_notification(f"✅ {description}", "success")
        st.rerun()
    else:
        pipeline.remove_step(step.action_id)
        st.error(f"❌ Failed: {result.get('error')}")


main()