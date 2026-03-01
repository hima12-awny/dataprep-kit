"""
Page 5: Feature Engineering
Column operations, aggregation, temporal features, and interaction features.
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
            <h1>⚙️ Feature Engineering</h1>
            <p>Create new features, aggregate, build temporal and interaction features</p>
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
        "📐 Column Ops",
        "📈 Aggregation",
        "⏱️ Temporal",
        "🔗 Interactions",
    ])

    with tab1:
        _render_recommendations(dataset)
    with tab2:
        _render_column_ops(dataset)
    with tab3:
        _render_aggregation(dataset)
    with tab4:
        _render_temporal(dataset)
    with tab5:
        _render_interactions(dataset)


# ══════════════════════════════════════════════════════════════
#  Helper: Action Signature for Deduplication
# ══════════════════════════════════════════════════════════════

def _action_signature(rec: dict) -> str:
    """
    Create a deduplication signature from a recommendation.
    Used to track which recommendations have already been approved,
    so analyzers don't re-recommend them after refresh.
    """
    action_type = rec.get("action_type", "")
    params = rec.get("parameters", {})

    columns = (
        params.get("columns")
        or params.get("agg_columns")
        or ([params["column"]] if params.get("column") else [])
    )
    if isinstance(columns, str):
        columns = [columns]
    columns = sorted(columns or [])

    operation = params.get("operation", "")

    # Include group_by for aggregation uniqueness
    group_by = params.get("group_by", "")
    if isinstance(group_by, list):
        group_by = "|".join(sorted(group_by))

    # Include expression / new_column for column_ops uniqueness
    extra = params.get("expression", "") or params.get("new_column", "")

    return f"{action_type}|{'|'.join(columns)}|{operation}|{group_by}|{extra}"


# ══════════════════════════════════════════════════════════════
#  Tab 1: Recommendations (Static + AI)
# ══════════════════════════════════════════════════════════════

def _render_recommendations(dataset):
    """Show feature engineering recommendations from static analysis and optional AI agent."""
    st.markdown("### 🤖 Smart Recommendations")
    st.markdown(
        "_Feature engineering suggestions based on your data's structure. "
        "Enable AI for deeper, context-aware suggestions._"
    )

    # ── Initialize trackers ───────────────────────────────
    if "feat_applied_actions" not in st.session_state:
        st.session_state["feat_applied_actions"] = set()

    # ── Inline success / error feedback ───────────────────
    last_applied = st.session_state.pop("feat_last_applied", None)
    if last_applied:
        st.success(f"✅ Successfully applied: **{last_applied}**")

    last_error = st.session_state.pop("feat_last_error", None)
    if last_error:
        st.error(f"❌ Action failed: {last_error}")

    # ── AI Configuration Status ───────────────────────────
    ai_configured = StateManager.is_ai_configured()
    ai_config = StateManager.get_ai_config()

    col_toggle, col_status = st.columns([1, 2])

    with col_toggle:
        use_ai = st.checkbox(
            "🤖 Include AI Recommendations",
            key="feat_use_ai",
            disabled=not ai_configured,
            help="Uses your configured AI model to suggest feature engineering actions",
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
    if st.button("🔄 Generate Recommendations", key="refresh_feat_recs"):
        st.session_state.pop("feat_recs_cache", None)
        st.session_state.pop("feat_rec_preview_data", None)
        st.session_state.pop("feat_recs_errors", None)
        st.session_state["feat_applied_actions"] = set()
        st.session_state["feat_force_generate"] = True
        st.rerun()

    # ── Generate / Cache Recommendations ──────────────────
    if "feat_recs_cache" not in st.session_state:
        engine = RecommendationEngine()

        force = st.session_state.pop("feat_force_generate", False)
        include_ai = use_ai and ai_configured and force
        effective_ai_config = ai_config if include_ai else None

        if effective_ai_config:
            with st.spinner(
                "🤖 AI is analyzing your dataset for feature engineering opportunities… "
                "This may take a moment."
            ):
                recs = engine.get_engineering_recommendations(
                    dataset, ai_config=effective_ai_config
                )
        else:
            recs = engine.get_engineering_recommendations(dataset)

        # Separate errors from actionable recommendations
        errors = [r for r in recs if r.get("action_type") == "error"]
        actionable = [r for r in recs if r.get("action_type") != "error"]

        # Filter out already-applied actions
        applied = st.session_state.get("feat_applied_actions", set())
        if applied:
            actionable = [
                r for r in actionable
                if _action_signature(r) not in applied
            ]

        st.session_state["feat_recs_cache"] = actionable
        st.session_state["feat_recs_ai_included"] = include_ai
        st.session_state["feat_recs_errors"] = errors

    recs = st.session_state.get("feat_recs_cache", [])
    ai_included = st.session_state.get("feat_recs_ai_included", False)
    errors = st.session_state.get("feat_recs_errors", [])

    # ── Display Errors ────────────────────────────────────
    if errors:
        for err in errors:
            author = err.get("author", "system")
            badge = "🤖 AI Agent" if author == "ai_agent" else "⚙️ Analyzer"
            st.warning(
                f"⚠️ {badge}: {err.get('description', 'Unknown error')}")

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
        title="Feature Engineering Recommendations",
        key_prefix="feat_rec",
    )


# ══════════════════════════════════════════════════════════════
#  Tab 2: Column Ops
# ══════════════════════════════════════════════════════════════

def _render_column_ops(dataset):
    """Column operations interface."""
    st.markdown("### 📐 Column Operations")
    df = dataset.df

    operation = st.selectbox(
        "Operation",
        ["create_expression", "combine_columns", "split_column",
         "rename_columns", "drop_columns"],
        key="colops_operation",
    )

    if operation == "create_expression":
        with st.form("create_expr_form"):
            new_column = st.text_input("New column name", key="ce_name")
            expression = st.text_input(
                "Expression",
                placeholder="e.g., col_A + col_B, col_A / col_B * 100",
                key="ce_expr",
                help="Reference column names directly. numpy (np) and pandas (pd) are available.",
            )

            st.markdown("**Available columns:**")
            st.caption(", ".join(f"`{c}`" for c in df.columns))

            submitted = st.form_submit_button(
                "✅ Create Column", width='stretch')

            if submitted and new_column and expression:
                _execute_user_action(
                    "column_ops",
                    {"operation": "create_expression",
                        "new_column": new_column, "expression": expression},
                    dataset,
                    f"Create column '{new_column}' = {expression}",
                )

    elif operation == "combine_columns":
        with st.form("combine_form"):
            columns = render_column_selector(
                df, key="combine_cols", label="Select columns to combine")
            separator = st.text_input(
                "Separator", value="_", key="combine_sep")
            new_column = st.text_input("New column name", key="combine_name")

            submitted = st.form_submit_button("✅ Combine", width='stretch')

            if submitted and columns and new_column:
                _execute_user_action(
                    "column_ops",
                    {"operation": "combine_columns", "columns": columns,
                     "separator": separator, "new_column": new_column},
                    dataset,
                    f"Combine [{', '.join(columns)}] into '{new_column}'",
                )

    elif operation == "split_column":
        with st.form("split_form"):
            column = st.selectbox(
                "Column to split", df.columns.tolist(), key="split_col")
            delimiter = st.text_input(
                "Delimiter", value=",", key="split_delim")
            max_splits = st.number_input(
                "Max splits (-1 for all)", value=-1, key="split_max")
            drop_original = st.checkbox(
                "Drop original column", value=False, key="split_drop")

            submitted = st.form_submit_button("✅ Split", width='stretch')

            if submitted:
                _execute_user_action(
                    "column_ops",
                    {"operation": "split_column", "column": column,
                     "delimiter": delimiter, "max_splits": max_splits,
                     "drop_original": drop_original},
                    dataset,
                    f"Split '{column}' by '{delimiter}'",
                )

    elif operation == "rename_columns":
        with st.form("rename_form"):
            st.markdown("**Current columns:**")
            rename_map = {}
            cols_per_row = 3
            col_list = df.columns.tolist()

            for i in range(0, len(col_list), cols_per_row):
                row_cols = st.columns(cols_per_row)
                for j, rc in enumerate(row_cols):
                    idx = i + j
                    if idx < len(col_list):
                        with rc:
                            new_name = st.text_input(
                                f"'{col_list[idx]}'",
                                value=col_list[idx],
                                key=f"rename_{idx}",
                            )
                            if new_name != col_list[idx]:
                                rename_map[col_list[idx]] = new_name

            submitted = st.form_submit_button("✅ Rename", width='stretch')

            if submitted and rename_map:
                _execute_user_action(
                    "column_ops",
                    {"operation": "rename_columns", "rename_map": rename_map},
                    dataset,
                    f"Rename {len(rename_map)} columns",
                )

    elif operation == "drop_columns":
        with st.form("drop_form"):
            columns = render_column_selector(
                df, key="drop_cols", label="Select columns to drop")
            submitted = st.form_submit_button(
                "✅ Drop Columns", width='stretch')

            if submitted and columns:
                _execute_user_action(
                    "column_ops",
                    {"operation": "drop_columns", "columns": columns},
                    dataset,
                    f"Drop columns: {columns}",
                )


# ══════════════════════════════════════════════════════════════
#  Tab 3: Aggregation
# ══════════════════════════════════════════════════════════════

def _render_aggregation(dataset):
    """Aggregation features interface."""
    st.markdown("### 📈 Aggregation Features")
    df = dataset.df

    operation = st.selectbox(
        "Operation",
        ["group_aggregate", "cumulative", "pivot", "melt"],
        key="agg_operation",
    )

    if operation == "group_aggregate":
        with st.form("group_agg_form"):
            group_by = render_column_selector(
                df, key="agg_group", label="Group by columns")
            agg_columns = render_column_selector(
                df, key="agg_cols", label="Columns to aggregate",
                type_filter="numeric",
            )
            agg_func = st.selectbox(
                "Aggregation function",
                ["mean", "sum", "median", "min", "max", "count", "std"],
                key="agg_func",
            )
            merge_back = st.checkbox(
                "Merge back to original DataFrame", value=True, key="agg_merge")

            submitted = st.form_submit_button("✅ Apply", width='stretch')

            if submitted and group_by and agg_columns:
                _execute_user_action(
                    "aggregation",
                    {"operation": "group_aggregate", "group_by": group_by,
                     "agg_columns": agg_columns, "agg_func": agg_func,
                     "merge_back": merge_back},
                    dataset,
                    f"Group by {group_by}, compute {agg_func} of {agg_columns}",
                )

    elif operation == "cumulative":
        with st.form("cumulative_form"):
            columns = render_column_selector(
                df, key="cum_cols", label="Select columns",
                type_filter="numeric",
            )
            cum_func = st.selectbox("Cumulative function", [
                                    "cumsum", "cumcount", "cumpct"], key="cum_func")
            group_by_col = st.selectbox(
                "Group by (optional)",
                ["None"] + df.columns.tolist(),
                key="cum_group",
            )

            submitted = st.form_submit_button("✅ Apply", width='stretch')

            if submitted and columns:
                params = {
                    "operation": "cumulative",
                    "columns": columns,
                    "cum_func": cum_func,
                }
                if group_by_col != "None":
                    params["group_by"] = group_by_col

                _execute_user_action(
                    "aggregation", params, dataset,
                    f"Cumulative {cum_func} on {columns}",
                )

    elif operation == "pivot":
        with st.form("pivot_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                index = st.selectbox(
                    "Index column", df.columns.tolist(), key="pivot_idx")
            with col2:
                pivot_column = st.selectbox(
                    "Pivot column", df.columns.tolist(), key="pivot_col")
            with col3:
                value_column = st.selectbox(
                    "Value column", df.columns.tolist(), key="pivot_val")

            agg_func = st.selectbox(
                "Agg function", ["mean", "sum", "count", "min", "max"], key="pivot_func")

            submitted = st.form_submit_button("✅ Pivot", width='stretch')

            if submitted:
                _execute_user_action(
                    "aggregation",
                    {"operation": "pivot", "index": index,
                     "pivot_column": pivot_column, "value_column": value_column,
                     "agg_func": agg_func},
                    dataset,
                    f"Pivot: index={index}, columns={pivot_column}, values={value_column}",
                )

    elif operation == "melt":
        with st.form("melt_form"):
            id_vars = render_column_selector(
                df, key="melt_id", label="ID columns (keep fixed)")
            value_vars = render_column_selector(
                df, key="melt_val", label="Value columns to unpivot (empty=all others)")
            var_name = st.text_input(
                "Variable name", value="variable", key="melt_var")
            value_name = st.text_input(
                "Value name", value="value", key="melt_valname")

            submitted = st.form_submit_button("✅ Melt", width='stretch')

            if submitted and id_vars:
                params = {
                    "operation": "melt",
                    "id_vars": id_vars,
                    "var_name": var_name,
                    "value_name": value_name,
                }
                if value_vars:
                    params["value_vars"] = value_vars

                _execute_user_action(
                    "aggregation", params, dataset,
                    f"Melt/Unpivot with id_vars={id_vars}",
                )


# ══════════════════════════════════════════════════════════════
#  Tab 4: Temporal
# ══════════════════════════════════════════════════════════════

def _render_temporal(dataset):
    """Temporal features interface."""
    st.markdown("### ⏱️ Temporal Features")
    df = dataset.df

    operation = st.selectbox(
        "Operation",
        ["lag", "lead", "rolling", "cyclical_encoding"],
        key="temp_operation",
    )

    if operation in ("lag", "lead"):
        with st.form(f"{operation}_form"):
            columns = render_column_selector(
                df, key=f"{operation}_cols", label="Select columns")
            periods = st.number_input(
                "Periods", min_value=1, value=1, key=f"{operation}_periods")
            group_by_col = st.selectbox(
                "Group by (optional)",
                ["None"] + df.columns.tolist(),
                key=f"{operation}_group",
            )

            submitted = st.form_submit_button("✅ Apply", width='stretch')

            if submitted and columns:
                params = {
                    "operation": operation,
                    "columns": columns,
                    "periods": periods,
                }
                if group_by_col != "None":
                    params["group_by"] = group_by_col

                _execute_user_action(
                    "temporal", params, dataset,
                    f"Create {operation}({periods}) for {columns}",
                )

    elif operation == "rolling":
        with st.form("rolling_form"):
            columns = render_column_selector(
                df, key="roll_cols", label="Select numeric columns",
                type_filter="numeric",
            )
            col1, col2 = st.columns(2)
            with col1:
                window = st.number_input(
                    "Window size", min_value=2, value=3, key="roll_window")
            with col2:
                rolling_func = st.selectbox(
                    "Function",
                    ["mean", "sum", "std", "min", "max", "median"],
                    key="roll_func",
                )

            group_by_col = st.selectbox(
                "Group by (optional)",
                ["None"] + df.columns.tolist(),
                key="roll_group",
            )

            submitted = st.form_submit_button("✅ Apply", width='stretch')

            if submitted and columns:
                params = {
                    "operation": "rolling",
                    "columns": columns,
                    "window": window,
                    "rolling_func": rolling_func,
                }
                if group_by_col != "None":
                    params["group_by"] = group_by_col

                _execute_user_action(
                    "temporal", params, dataset,
                    f"Rolling {rolling_func}(window={window}) for {columns}",
                )

    elif operation == "cyclical_encoding":
        with st.form("cyclical_form"):
            column = st.selectbox(
                "Select column", df.columns.tolist(), key="cyc_col")
            max_value = st.number_input(
                "Max cycle value",
                min_value=1, value=24,
                help="e.g., 24 for hours, 7 for weekdays, 12 for months",
                key="cyc_max",
            )
            drop_original = st.checkbox(
                "Drop original column", value=False, key="cyc_drop")

            submitted = st.form_submit_button("✅ Apply", width='stretch')

            if submitted:
                _execute_user_action(
                    "temporal",
                    {"operation": "cyclical_encoding", "column": column,
                     "max_value": max_value, "drop_original": drop_original},
                    dataset,
                    f"Cyclical encoding of '{column}' (max={max_value})",
                )


# ══════════════════════════════════════════════════════════════
#  Tab 5: Interactions
# ══════════════════════════════════════════════════════════════

def _render_interactions(dataset):
    """Interaction features interface."""
    st.markdown("### 🔗 Interaction Features")
    df = dataset.df

    operation = st.selectbox(
        "Operation",
        ["polynomial", "pairwise_ratio", "pairwise_difference",
         "pairwise_product", "cross_categorical"],
        key="inter_operation",
    )

    is_numeric_op = operation in (
        "polynomial", "pairwise_ratio", "pairwise_difference", "pairwise_product")

    with st.form("interaction_form"):
        if is_numeric_op:
            columns = render_column_selector(
                df, key="inter_cols", label="Select numeric columns",
                type_filter="numeric",
            )
        else:
            columns = render_column_selector(
                df, key="inter_cols_cat", label="Select categorical columns",
                type_filter="categorical",
            )

        degree = 2
        if operation == "polynomial":
            degree = st.number_input(
                "Polynomial degree", min_value=2, max_value=4, value=2, key="inter_deg")

        submitted = st.form_submit_button("✅ Apply", width='stretch')

        if submitted and columns:
            params = {
                "operation": operation,
                "columns": columns,
            }
            if operation == "polynomial":
                params["degree"] = degree

            _execute_user_action(
                "interaction", params, dataset,
                f"{operation} features for {columns}",
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
        applied = st.session_state.get("feat_applied_actions", set())
        applied.add(sig)
        st.session_state["feat_applied_actions"] = applied

        # ── Remove ONLY this action from cache ────────────────
        recs = st.session_state.get("feat_recs_cache", [])
        st.session_state["feat_recs_cache"] = [
            r for r in recs
            if r.get("action_id") != recommendation.get("action_id")
        ]

        # ── Store success feedback ────────────────────────────
        desc = recommendation.get("description", "Action")
        rows_before = result.get("rows_before", "?")
        rows_after = result.get("rows_after", "?")
        cols_before = result.get("cols_before", "?")
        cols_after = result.get("cols_after", "?")
        st.session_state["feat_last_applied"] = (
            f"{desc}  (rows: {rows_before} → {rows_after}, "
            f"cols: {cols_before} → {cols_after})"
        )

        st.session_state.pop("feat_rec_preview_data", None)

        StateManager.add_notification(
            f"✅ Applied: {desc}", "success"
        )
        st.rerun()
    else:
        error_msg = result.get("error", "Unknown error")
        st.session_state["feat_last_error"] = error_msg
        pipeline.remove_step(step.action_id)
        st.rerun()


def _reject_action(recommendation: dict):
    """Dismiss a recommendation.
    Removes ONLY this action from cache — preserves all others.
    """
    recs = st.session_state.get("feat_recs_cache", [])
    st.session_state["feat_recs_cache"] = [
        r for r in recs
        if r.get("action_id") != recommendation.get("action_id")
    ]
    st.session_state.pop("feat_rec_preview_data", None)
    st.rerun()


def _preview_action(recommendation: dict, dataset):
    """Preview a recommended action — stores keyed by action_id for inline rendering."""
    action_id = recommendation.get("action_id", "unknown")
    step = PipelineStep.from_dict(recommendation)
    pipeline = StateManager.get_pipeline()
    preview = pipeline.preview_step(step, dataset)
    st.session_state[f"feat_rec_preview_{action_id}"] = preview


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
        applied = st.session_state.get("feat_applied_actions", set())
        applied.add(sig)
        st.session_state["feat_applied_actions"] = applied

        # Mark stale but do NOT delete cache
        st.session_state["feat_recs_stale"] = True

        st.session_state.pop("feat_rec_preview_data", None)

        st.session_state["feat_last_applied"] = description
        StateManager.add_notification(f"✅ {description}", "success")
        st.rerun()
    else:
        pipeline.remove_step(step.action_id)
        st.error(f"❌ Failed: {result.get('error')}")


main()
