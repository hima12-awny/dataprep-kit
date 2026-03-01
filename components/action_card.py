"""
Renders a single recommendation action card with approve/preview/edit/reject controls.
Edit mode provides type-aware UI widgets for every parameter.
Parameters displayed as clean, human-readable UI instead of raw JSON.
"""

import streamlit as st
from typing import Dict, Callable, Optional, List, Any


# ══════════════════════════════════════════════════════════════
#  Parameter Schema Registry
# ══════════════════════════════════════════════════════════════

PARAM_SCHEMAS = {
    # ── Cleaning ──────────────────────────────────────────
    "handle_missing": [
        {"key": "columns", "label": "Columns", "type": "multiselect_columns"},
        {"key": "strategy", "label": "Strategy", "type": "select",
         "options": ["mean", "median", "mode", "constant", "forward_fill",
                     "backward_fill", "drop_rows", "drop_columns", "group_based"]},
        {"key": "fill_value", "label": "Fill Value (constant only)", "type": "text",
         "show_if": {"strategy": "constant"}},
        {"key": "group_by", "label": "Group By Column", "type": "select_column",
         "show_if": {"strategy": "group_based"}},
        {"key": "group_strategy", "label": "Group Strategy", "type": "select",
         "options": ["mean", "median", "mode"],
         "show_if": {"strategy": "group_based"}},
    ],
    "handle_duplicates": [
        {"key": "subset",
            "label": "Subset Columns (empty = all)", "type": "multiselect_columns"},
        {"key": "keep", "label": "Keep", "type": "select",
         "options": ["first", "last", "none"]},
    ],
    "handle_outliers": [
        {"key": "columns", "label": "Columns",
            "type": "multiselect_columns", "filter": "numeric"},
        {"key": "method", "label": "Method", "type": "select",
         "options": ["iqr", "zscore", "percentile"]},
        {"key": "threshold", "label": "Threshold",
            "type": "number", "default": 1.5, "step": 0.1},
        {"key": "behavior", "label": "Behavior", "type": "select",
         "options": ["clip", "remove", "flag"]},
    ],
    "text_cleaning": [
        {"key": "columns", "label": "Columns",
            "type": "multiselect_columns", "filter": "text"},
        {"key": "operations", "label": "Operations", "type": "multiselect",
         "options": ["trim_whitespace", "lowercase", "uppercase", "titlecase",
                     "remove_special_chars", "remove_punctuation",
                     "collapse_whitespace", "strip_html"]},
    ],
    "inconsistency": [
        {"key": "columns", "label": "Columns", "type": "multiselect_columns"},
        {"key": "operation", "label": "Operation", "type": "select",
         "options": ["value_mapping", "merge_rare_categories", "standardize_values"]},
        {"key": "threshold", "label": "Frequency Threshold", "type": "number",
         "default": 0.01, "step": 0.01,
         "show_if": {"operation": "merge_rare_categories"}},
        {"key": "replacement", "label": "Replacement Value", "type": "text",
         "show_if": {"operation": "merge_rare_categories"}},
    ],
    # ── Conversion ────────────────────────────────────────
    "type_casting": [
        {"key": "conversions", "label": "Conversions", "type": "conversions_editor"},
        {"key": "errors", "label": "Error Handling", "type": "select",
         "options": ["coerce", "raise", "ignore"]},
        {"key": "date_format",
            "label": "Date Format (optional)", "type": "text"},
    ],
    "datetime_ops": [
        {"key": "operation", "label": "Operation", "type": "select",
         "options": ["extract_components", "date_diff", "to_unix_timestamp", "from_unix_timestamp"]},
        {"key": "column", "label": "Source Column", "type": "select_column"},
        {"key": "components", "label": "Components", "type": "multiselect",
         "options": ["year", "month", "day", "weekday", "day_name", "quarter",
                     "week", "is_weekend", "is_month_start", "is_month_end", "hour", "minute"],
         "show_if": {"operation": "extract_components"}},
        {"key": "column2", "label": "Second Column", "type": "select_column",
         "show_if": {"operation": "date_diff"}},
        {"key": "unit", "label": "Unit", "type": "select",
         "options": ["days", "hours", "seconds"],
         "show_if": {"operation": "date_diff"}},
    ],
    "numeric_transform": [
        {"key": "columns", "label": "Columns",
            "type": "multiselect_columns", "filter": "numeric"},
        {"key": "operation", "label": "Operation", "type": "select",
         "options": ["log", "log1p", "sqrt", "square", "abs", "round",
                     "normalize", "standardize", "robust_scale", "clip", "binning"]},
        {"key": "overwrite", "label": "Overwrite Original",
            "type": "checkbox", "default": True},
        {"key": "n_bins", "label": "Number of Bins", "type": "number",
         "default": 5, "step": 1, "show_if": {"operation": "binning"}},
    ],
    "encoding": [
        {"key": "columns", "label": "Columns",
            "type": "multiselect_columns", "filter": "categorical"},
        {"key": "method", "label": "Method", "type": "select",
         "options": ["onehot", "label", "frequency", "ordinal", "binary", "target"]},
        {"key": "drop_first", "label": "Drop First Dummy", "type": "checkbox", "default": False,
         "show_if": {"method": "onehot"}},
        {"key": "drop_original", "label": "Drop Original",
            "type": "checkbox", "default": True},
        {"key": "target_column", "label": "Target Column", "type": "select_column",
         "filter": "numeric", "show_if": {"method": "target"}},
    ],
    # ── Feature Engineering ───────────────────────────────
    "column_ops": [
        {"key": "operation", "label": "Operation", "type": "select",
         "options": ["create_expression", "combine_columns", "split_column",
                     "rename_columns", "drop_columns"]},
        {"key": "columns", "label": "Columns", "type": "multiselect_columns",
         "show_if": {"operation": ["combine_columns", "drop_columns"]}},
        {"key": "column", "label": "Source Column", "type": "select_column",
         "show_if": {"operation": "split_column"}},
        {"key": "new_column", "label": "New Column Name", "type": "text",
         "show_if": {"operation": ["create_expression", "combine_columns"]}},
        {"key": "expression", "label": "Expression", "type": "text",
         "show_if": {"operation": "create_expression"}},
        {"key": "separator", "label": "Separator", "type": "text",
         "show_if": {"operation": "combine_columns"}},
        {"key": "delimiter", "label": "Delimiter", "type": "text",
         "show_if": {"operation": "split_column"}},
    ],
    "aggregation": [
        {"key": "operation", "label": "Operation", "type": "select",
         "options": ["group_aggregate", "cumulative", "pivot", "melt"]},
        {"key": "group_by", "label": "Group By", "type": "multiselect_columns",
         "show_if": {"operation": ["group_aggregate", "cumulative"]}},
        {"key": "agg_columns", "label": "Aggregate Columns", "type": "multiselect_columns",
         "filter": "numeric", "show_if": {"operation": "group_aggregate"}},
        {"key": "agg_func", "label": "Aggregation Function", "type": "select",
         "options": ["mean", "sum", "median", "min", "max", "count", "std"],
         "show_if": {"operation": ["group_aggregate", "pivot"]}},
        {"key": "merge_back", "label": "Merge Back", "type": "checkbox", "default": True,
         "show_if": {"operation": "group_aggregate"}},
    ],
    "temporal": [
        {"key": "operation", "label": "Operation", "type": "select",
         "options": ["lag", "lead", "rolling", "cyclical_encoding"]},
        {"key": "columns", "label": "Columns", "type": "multiselect_columns",
         "show_if": {"operation": ["lag", "lead", "rolling"]}},
        {"key": "column", "label": "Column", "type": "select_column",
         "show_if": {"operation": "cyclical_encoding"}},
        {"key": "periods", "label": "Periods", "type": "number", "default": 1, "step": 1,
         "show_if": {"operation": ["lag", "lead"]}},
        {"key": "window", "label": "Window Size", "type": "number", "default": 3, "step": 1,
         "show_if": {"operation": "rolling"}},
        {"key": "rolling_func", "label": "Rolling Function", "type": "select",
         "options": ["mean", "sum", "std", "min", "max", "median"],
         "show_if": {"operation": "rolling"}},
        {"key": "max_value", "label": "Max Cycle Value", "type": "number",
         "default": 24, "step": 1,
         "show_if": {"operation": "cyclical_encoding"}},
    ],
    "interaction": [
        {"key": "operation", "label": "Operation", "type": "select",
         "options": ["polynomial", "pairwise_ratio", "pairwise_difference",
                     "pairwise_product", "cross_categorical"]},
        {"key": "columns", "label": "Columns", "type": "multiselect_columns"},
        {"key": "degree", "label": "Polynomial Degree", "type": "number",
         "default": 2, "step": 1,
         "show_if": {"operation": "polynomial"}},
    ],
}


# ══════════════════════════════════════════════════════════════
#  Human-Readable Labels
# ══════════════════════════════════════════════════════════════

PARAM_LABELS = {
    # Keys → friendly labels
    "columns": "📋 Columns",
    "column": "📋 Column",
    "column2": "📋 Second Column",
    "subset": "📋 Subset Columns",
    "strategy": "🎯 Strategy",
    "method": "🎯 Method",
    "operation": "🔧 Operation",
    "behavior": "⚡ Behavior",
    "fill_value": "✏️ Fill Value",
    "threshold": "📏 Threshold",
    "operations": "🔧 Operations",
    "conversions": "🔄 Type Conversions",
    "errors": "⚠️ Error Handling",
    "date_format": "📅 Date Format",
    "components": "📦 Components",
    "unit": "📐 Unit",
    "overwrite": "♻️ Overwrite Original",
    "n_bins": "📊 Number of Bins",
    "drop_first": "🗑️ Drop First Dummy",
    "drop_original": "🗑️ Drop Original",
    "target_column": "🎯 Target Column",
    "new_column": "✨ New Column",
    "expression": "🧮 Expression",
    "separator": "✂️ Separator",
    "delimiter": "✂️ Delimiter",
    "rename_map": "📝 Rename Map",
    "group_by": "👥 Group By",
    "agg_columns": "📊 Aggregate Columns",
    "agg_func": "📈 Aggregation Function",
    "merge_back": "🔗 Merge Back",
    "periods": "🔢 Periods",
    "window": "📐 Window Size",
    "rolling_func": "📈 Rolling Function",
    "max_value": "🔄 Max Cycle Value",
    "degree": "📐 Polynomial Degree",
    "keep": "📌 Keep",
    "mapping": "📝 Value Mapping",
    "replacement": "🔄 Replacement",
    "cum_func": "📈 Cumulative Function",
    "id_vars": "🔑 ID Columns",
    "value_vars": "📊 Value Columns",
    "var_name": "📝 Variable Name",
    "value_name": "📝 Value Name",
}

STRATEGY_DESCRIPTIONS = {
    "mean": "Fill with column mean",
    "median": "Fill with column median",
    "mode": "Fill with most frequent value",
    "constant": "Fill with a fixed value",
    "forward_fill": "Propagate last valid value forward",
    "backward_fill": "Propagate next valid value backward",
    "drop_rows": "Remove rows with missing values",
    "drop_columns": "Remove columns entirely",
    "group_based": "Fill based on group statistics",
    "iqr": "Interquartile Range method",
    "zscore": "Z-Score method",
    "percentile": "Percentile-based method",
    "clip": "Cap values at boundaries",
    "remove": "Remove outlier rows",
    "flag": "Add a boolean flag column",
    "onehot": "One-Hot (dummy) encoding",
    "label": "Integer label encoding",
    "frequency": "Frequency-based encoding",
    "ordinal": "Ordinal position encoding",
    "binary": "Binary encoding",
    "target": "Target-mean encoding",
    "log": "Natural logarithm",
    "log1p": "Log(1 + x) — safe for zeros",
    "sqrt": "Square root",
    "square": "Square (x²)",
    "abs": "Absolute value",
    "round": "Round to decimal places",
    "normalize": "Min-max to [0, 1]",
    "standardize": "Z-score (mean=0, std=1)",
    "robust_scale": "Median/IQR scaling",
    "binning": "Bin into equal-width buckets",
    "coerce": "Invalid → NaN",
    "raise": "Invalid → Error",
    "ignore": "Invalid → Keep original",
}


# ══════════════════════════════════════════════════════════════
#  Main Card Renderer
# ══════════════════════════════════════════════════════════════

def render_action_card(
    recommendation: Dict,
    on_approve: Optional[Callable] = None,
    on_reject: Optional[Callable] = None,
    on_preview: Optional[Callable] = None,
    show_controls: bool = True,
    key_prefix: str = "",
):
    """Render a single action recommendation card with Edit mode."""
    action_id = recommendation.get("action_id", "unknown")
    action_type = recommendation.get("action_type", "unknown")
    description = recommendation.get("description", "No description")
    author = recommendation.get("author", "user")
    priority = recommendation.get("priority", "low")
    reason = recommendation.get("reason", "")

    icon = _action_type_icon(action_type)
    badge_label, badge_color = _author_badge(author)

    edit_key = f"{key_prefix}_edit_{action_id}"
    is_editing = st.session_state.get(edit_key, False)

    # ── Card Container ────────────────────────────────────
    with st.container(border=True):

        st.markdown(f"<span style='background:{badge_color}; color:white; padding:2px 8px; border-radius:10px; font-size:0.9rem; "
                    f"font-weight:700; white-space:nowrap; margin-right: 5px'>{badge_label}</span> "
                    
                    f"Recommend to **{icon} {description}**", unsafe_allow_html=True)



        # ── EDIT MODE ─────────────────────────────────────
        if is_editing:
            _render_edit_mode(recommendation, key_prefix,
                              action_id, edit_key, on_approve)

        # ── NORMAL MODE ───────────────────────────────────
        else:
            if reason:
                with st.expander("💡 Why this recommendation?", expanded=False):
                    st.markdown(f"_{reason}_")
                    st.caption(
                        f"Action type: `{action_type}` | Priority: **{priority}**")

                    params = recommendation.get("parameters", {})
                    if params:
                        st.markdown("---")
                        _render_params_display(action_type, params)

            if show_controls:
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    if st.button("✅ Approve", key=f"{key_prefix}_approve_{action_id}",
                                 width='stretch'):
                        if on_approve:
                            on_approve(recommendation)

                with col2:
                    if st.button("👁️ Preview", key=f"{key_prefix}_preview_{action_id}",
                                 width='stretch'):
                        if on_preview:
                            on_preview(recommendation)

                with col3:
                    if st.button("✏️ Edit", key=f"{key_prefix}_editbtn_{action_id}",
                                 width='stretch'):
                        st.session_state[edit_key] = True
                        st.rerun()

                with col4:
                    if st.button("❌ Dismiss", key=f"{key_prefix}_reject_{action_id}",
                                 width='stretch'):
                        if on_reject:
                            on_reject(recommendation)


# ══════════════════════════════════════════════════════════════
#  Parameter Display (Read-Only, Human-Readable)
# ══════════════════════════════════════════════════════════════

def _render_params_display(action_type: str, params: Dict):
    """Render parameters as clean, human-readable UI elements."""

    st.markdown("**⚙️ Parameters:**")

    for key, value in params.items():
        if value is None or value == "" or value == []:
            continue

        label = PARAM_LABELS.get(key, key.replace("_", " ").title())

        # ── Lists (columns, operations, components) ───────
        if isinstance(value, list):
            pills = "  ".join(
                [f"`{v}`" for v in value]
            )
            st.markdown(f"{label}: {pills}")

        # ── Dicts (conversions, mapping, rename_map) ──────
        elif isinstance(value, dict):
            if key == "conversions":
                _render_conversions_display(value)
            elif key == "mapping":
                _render_mapping_display(value)
            elif key == "rename_map":
                _render_rename_display(value)
            else:
                # Generic dict display
                for dk, dv in value.items():
                    st.markdown(f"  - `{dk}` → `{dv}`")

        # ── Booleans ──────────────────────────────────────
        elif isinstance(value, bool):
            icon = "✅" if value else "❌"
            st.markdown(f"{label}: {icon} {'Yes' if value else 'No'}")

        # ── Numbers ───────────────────────────────────────
        elif isinstance(value, (int, float)):
            st.markdown(f"{label}: **{value}**")

        # ── Strings (strategy, method, operation, etc.) ───
        elif isinstance(value, str):
            hint = STRATEGY_DESCRIPTIONS.get(value, "")
            if hint:
                st.markdown(f"{label}: **{value}** — _{hint}_")
            else:
                st.markdown(f"{label}: **{value}**")


def _render_conversions_display(conversions: Dict):
    """Render type_casting conversions as a clean table."""
    st.markdown("🔄 **Type Conversions:**")
    for col_name, target_type in conversions.items():
        st.markdown(f"  - `{col_name}` → **{target_type}**")


def _render_mapping_display(mapping: Dict):
    """Render value mapping as a clean table."""
    st.markdown("📝 **Value Mapping:**")
    for old_val, new_val in mapping.items():
        st.markdown(f"  - `{old_val}` → `{new_val}`")


def _render_rename_display(rename_map: Dict):
    """Render rename mapping as a clean table."""
    st.markdown("📝 **Rename Map:**")
    for old_name, new_name in rename_map.items():
        st.markdown(f"  - `{old_name}` → `{new_name}`")


# ══════════════════════════════════════════════════════════════
#  Edit Mode Renderer
# ══════════════════════════════════════════════════════════════

def _render_edit_mode(recommendation: Dict, key_prefix: str, action_id: str, edit_key: str,
                      on_approve: Optional[Callable]):
    """Render type-aware UI widgets for editing action parameters."""

    action_type = recommendation.get("action_type", "")
    params = recommendation.get("parameters", {})
    schema = PARAM_SCHEMAS.get(action_type, [])

    df_columns = _get_df_columns()
    numeric_cols = _get_df_columns("numeric")
    text_cols = _get_df_columns("text")
    cat_cols = _get_df_columns("categorical")

    st.markdown("---")
    st.markdown("##### ✏️ Edit Parameters")

    # Description editor
    new_desc = st.text_input(
        "Description",
        value=recommendation.get("description", ""),
        key=f"{key_prefix}_edesc_{action_id}",
    )

    # Priority editor
    current_priority = recommendation.get("priority", "medium")
    new_priority = st.select_slider(
        "Priority",
        options=["low", "medium", "high"],
        value=current_priority,
        key=f"{key_prefix}_epri_{action_id}",
    )

    # Dynamic parameter widgets
    new_params = {}
    current_vals = dict(params)

    for field in schema:
        fkey = field["key"]
        ftype = field["type"]
        flabel = field["label"]
        widget_key = f"{key_prefix}_e_{fkey}_{action_id}"

        if not _should_show(field, current_vals):
            if fkey in params:
                new_params[fkey] = params[fkey]
            continue

        current_val = params.get(fkey)

        if ftype == "multiselect_columns":
            col_filter = field.get("filter", "")
            if col_filter == "numeric":
                options = numeric_cols
            elif col_filter == "text":
                options = text_cols
            elif col_filter == "categorical":
                options = cat_cols
            else:
                options = df_columns

            default = current_val if isinstance(current_val, list) else []
            default = [c for c in default if c in options]
            val = st.multiselect(flabel, options=options,
                                 default=default, key=widget_key)
            new_params[fkey] = val

        elif ftype == "select_column":
            col_filter = field.get("filter", "")
            if col_filter == "numeric":
                options = numeric_cols
            elif col_filter == "text":
                options = text_cols
            elif col_filter == "categorical":
                options = cat_cols
            else:
                options = df_columns

            default_idx = 0
            if current_val and current_val in options:
                default_idx = options.index(current_val)
            if options:
                val = st.selectbox(flabel, options=options,
                                   index=default_idx, key=widget_key)
                new_params[fkey] = val

        elif ftype == "select":
            options = field.get("options", [])
            default_idx = 0
            if current_val and current_val in options:
                default_idx = options.index(current_val)
            val = st.selectbox(flabel, options=options,
                               index=default_idx, key=widget_key)
            new_params[fkey] = val
            current_vals[fkey] = val

        elif ftype == "multiselect":
            options = field.get("options", [])
            default = current_val if isinstance(current_val, list) else []
            default = [v for v in default if v in options]
            val = st.multiselect(flabel, options=options,
                                 default=default, key=widget_key)
            new_params[fkey] = val

        elif ftype == "number":
            default = current_val if current_val is not None else field.get(
                "default", 0)
            step = field.get("step", 0.1)
            val = st.number_input(flabel, value=float(
                default), step=float(step), key=widget_key)
            new_params[fkey] = int(val) if step == 1 else val

        elif ftype == "checkbox":
            default = current_val if current_val is not None else field.get(
                "default", False)
            val = st.checkbox(flabel, value=bool(default), key=widget_key)
            new_params[fkey] = val

        elif ftype == "text":
            default = str(current_val) if current_val is not None else ""
            val = st.text_input(flabel, value=default, key=widget_key)
            if val:
                new_params[fkey] = val

        elif ftype == "conversions_editor":
            _render_conversions_editor(
                params, new_params, widget_key, df_columns)

    # Save / Cancel
    st.markdown("---")
    col_save, col_preview, col_cancel = st.columns(3)

    with col_save:
        if st.button("💾 Save & Approve", key=f"{key_prefix}_esave_{action_id}",
                     type="primary", width='stretch'):
            recommendation["description"] = new_desc
            recommendation["priority"] = new_priority
            cleaned = {k: v for k, v in new_params.items()
                       if v is not None and v != "" and v != []}
            recommendation["parameters"] = cleaned
            st.session_state[edit_key] = False
            if on_approve:
                on_approve(recommendation)

    with col_preview:
        if st.button("💾 Save Edits", key=f"{key_prefix}_esaveonly_{action_id}",
                     width='stretch'):
            recommendation["description"] = new_desc
            recommendation["priority"] = new_priority
            cleaned = {k: v for k, v in new_params.items()
                       if v is not None and v != "" and v != []}
            recommendation["parameters"] = cleaned
            st.session_state[edit_key] = False
            st.success("✅ Edits saved!")
            st.rerun()

    with col_cancel:
        if st.button("❌ Cancel", key=f"{key_prefix}_ecancel_{action_id}",
                     width='stretch'):
            st.session_state[edit_key] = False
            st.rerun()


# ══════════════════════════════════════════════════════════════
#  Conversions Editor (for type_casting edit mode)
# ══════════════════════════════════════════════════════════════

def _render_conversions_editor(params: Dict, new_params: Dict, widget_key: str,
                               df_columns: List[str]):
    """Render editable rows for the type_casting conversions dict."""
    st.markdown("**Column → Type Conversions:**")
    conversions = params.get("conversions", {})
    type_options = ["int64", "float64", "string",
                    "bool", "datetime64[ns]", "category"]

    new_conversions = {}
    for i, (col, target_type) in enumerate(conversions.items()):
        c1, c2 = st.columns(2)
        with c1:
            col_options = df_columns
            col_idx = col_options.index(col) if col in col_options else 0
            new_col = st.selectbox(
                f"Column {i+1}", options=col_options, index=col_idx,
                key=f"{widget_key}_conv_col_{i}",
            )
        with c2:
            type_idx = type_options.index(
                target_type) if target_type in type_options else 0
            new_type = st.selectbox(
                f"Target Type {i+1}", options=type_options, index=type_idx,
                key=f"{widget_key}_conv_type_{i}",
            )
        new_conversions[new_col] = new_type

    new_params["conversions"] = new_conversions


# ══════════════════════════════════════════════════════════════
#  Condition Evaluation
# ══════════════════════════════════════════════════════════════

def _should_show(field: Dict, current_vals: Dict) -> bool:
    """Check if a field should be visible based on show_if conditions."""
    show_if = field.get("show_if")
    if not show_if:
        return True

    for cond_key, cond_val in show_if.items():
        actual = current_vals.get(cond_key, "")
        if isinstance(cond_val, list):
            if actual not in cond_val:
                return False
        else:
            if actual != cond_val:
                return False
    return True


# ══════════════════════════════════════════════════════════════
#  DataFrame Column Helpers
# ══════════════════════════════════════════════════════════════

def _get_df_columns(filter_type: str = "") -> List[str]:
    """Get column names from the current dataset in session state."""
    try:
        from core.state import StateManager
        dataset = StateManager.get_dataset()
        if dataset is None:
            return []
        df = dataset.df

        if filter_type == "numeric":
            return df.select_dtypes(include=["number"]).columns.tolist()
        elif filter_type == "text":
            return df.select_dtypes(include=["object", "string"]).columns.tolist()
        elif filter_type == "categorical":
            return df.select_dtypes(include=["object", "category"]).columns.tolist()
        else:
            return df.columns.tolist()
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════
#  Visual Helpers
# ══════════════════════════════════════════════════════════════

def _author_badge(author: str):
    """Return (label, color) for the author badge."""
    badges = {
        "ai_static": ("Static AI", "#10B981"),
        "ai_agent": ("🤖 AI Agent", "#8B5CF6"),
        "both": ("⚡ Both", "#F59E0B"),
        "system": ("⚙️ System", "#6B7280"),
        "user": ("👤 User", "#3B82F6"),
    }
    return badges.get(author, (author.capitalize(), "#6B7280"))


def _action_type_icon(action_type: str) -> str:
    """Return an emoji icon for the action type."""
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


def _priority_color(priority: str) -> str:
    """Return color hex for a priority level."""
    return {
        "high": "#EF4444",
        "medium": "#F59E0B",
        "low": "#10B981",
    }.get(priority, "#10B981")
