"""
Page 3: Data Cleaning
Handle missing values, duplicates, outliers, and text issues.
Shows static + AI recommendations with approve/preview/reject controls.
"""

import streamlit as st

from core.state import StateManager
from core.pipeline import PipelineStep
from recommendations.engine import RecommendationEngine
from components.metrics_bar import render_metrics_bar
from components.action_list import render_action_list
from components.preview_table import render_preview_table
from components.column_selector import render_column_selector
from config.registry import ActionRegistry
from utils.id_generator import generate_action_id


def main():

    st.markdown(
        """
        <div class="page-header">
            <h1>🧹 Data Cleaning</h1>
            <p>Handle missing values, duplicates, outliers, and text inconsistencies</p>
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
        "🕳️ Missing Values",
        "👥 Duplicates",
        "📊 Outliers",
        "✏️ Text Cleaning",
    ])

    with tab1:
        _render_recommendations(dataset)
    with tab2:
        _render_missing_values(dataset)
    with tab3:
        _render_duplicates(dataset)
    with tab4:
        _render_outliers(dataset)
    with tab5:
        _render_text_cleaning(dataset)


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
    columns = sorted(columns or [])

    strategy = (
        params.get("strategy")
        or params.get("method")
        or params.get("operation")
        or ""
    )
    behavior = params.get("behavior", "")

    return f"{action_type}|{'|'.join(columns)}|{strategy}|{behavior}"


# ══════════════════════════════════════════════════════════════
#  Tab 1: Recommendations (Static + AI)
# ══════════════════════════════════════════════════════════════

def _render_recommendations(dataset):
    """Show cleaning recommendations from static analysis and optional AI agent."""
    st.markdown("### 🤖 Smart Recommendations")
    st.markdown(
        "_Detected automatically from your data profile. "
        "Enable AI for deeper, context-aware suggestions._"
    )

    # ── Initialize trackers ───────────────────────────────
    if "clean_applied_actions" not in st.session_state:
        st.session_state["clean_applied_actions"] = set()

    # ── Inline success / error feedback ───────────────────
    last_applied = st.session_state.pop("clean_last_applied", None)
    if last_applied:
        st.success(f"✅ Successfully applied: **{last_applied}**")

    last_error = st.session_state.pop("clean_last_error", None)
    if last_error:
        st.error(f"❌ Action failed: {last_error}")

    # ── AI Configuration Status ───────────────────────────
    ai_configured = StateManager.is_ai_configured()
    ai_config = StateManager.get_ai_config()

    col_toggle, col_status = st.columns([1, 2])

    with col_toggle:
        use_ai = st.checkbox(
            "🤖 Include AI Recommendations",
            key="clean_use_ai",
            disabled=not ai_configured,
            help="Uses your configured AI model to suggest additional cleaning actions",
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
    # This is the ONLY place that triggers full regeneration (including AI)
    if st.button("🔄 Generate Recommendations", key="refresh_clean_recs"):
        st.session_state.pop("clean_recs_cache", None)
        st.session_state.pop("clean_rec_preview_data", None)
        st.session_state.pop("clean_recs_errors", None)
        st.session_state["clean_applied_actions"] = set()
        st.session_state["clean_force_generate"] = True
        st.rerun()

    # ── Generate / Cache Recommendations ──────────────────
    # Only generated on first visit OR explicit button click
    if "clean_recs_cache" not in st.session_state:
        engine = RecommendationEngine()

        # Include AI only when user explicitly clicked Generate
        force = st.session_state.pop("clean_force_generate", False)
        include_ai = use_ai and ai_configured and force
        effective_ai_config = ai_config if include_ai else None

        if effective_ai_config:
            with st.spinner(
                "🤖 AI is analyzing your dataset for cleaning issues… "
                "This may take a moment."
            ):
                recs = engine.get_cleaning_recommendations(
                    dataset, ai_config=effective_ai_config
                )
        else:
            recs = engine.get_cleaning_recommendations(dataset)

        # Separate errors from actionable recommendations
        errors = [r for r in recs if r.get("action_type") == "error"]
        actionable = [r for r in recs if r.get("action_type") != "error"]

        # Filter out already-applied actions
        applied = st.session_state.get("clean_applied_actions", set())
        if applied:
            actionable = [
                r for r in actionable
                if _action_signature(r) not in applied
            ]

        st.session_state["clean_recs_cache"] = actionable
        st.session_state["clean_recs_ai_included"] = include_ai
        st.session_state["clean_recs_errors"] = errors

    recs = st.session_state.get("clean_recs_cache", [])
    ai_included = st.session_state.get("clean_recs_ai_included", False)
    errors = st.session_state.get("clean_recs_errors", [])

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
        on_preview=lambda r: _preview_action(r, dataset, "clean_rec"),
        title="Cleaning Recommendations",
        key_prefix="clean_rec",
    )


# ══════════════════════════════════════════════════════════════
#  Tab 2: Missing Values
# ══════════════════════════════════════════════════════════════

def _render_missing_values(dataset):
    """Manual missing value handling interface."""
    st.markdown("### 🕳️ Handle Missing Values")
    df = dataset.df

    missing = df.isna().sum()
    missing = missing[missing > 0]

    if len(missing) == 0:
        st.success("✅ No missing values found!")
        return

    st.dataframe(
        missing.reset_index().rename(
            columns={"index": "Column", 0: "Missing Count"}),
        width='stretch',
        hide_index=True,
    )

    with st.form("missing_form"):
        columns = render_column_selector(
            df, key="missing_cols", label="Select columns to handle"
        )

        strategy = st.selectbox(
            "Strategy",
            ["mean", "median", "mode", "constant", "forward_fill",
             "backward_fill", "drop_rows", "drop_columns"],
        )

        fill_value = None
        if strategy == "constant":
            fill_value = st.text_input("Fill value")

        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button(
                "✅ Apply", width='stretch')
        with col2:
            preview_btn = st.form_submit_button(
                "👁️ Preview", width='stretch')

        if submitted and columns:
            params = {
                "columns": columns,
                "strategy": strategy,
                "fill_value": fill_value,
            }
            _execute_user_action(
                "handle_missing", params, dataset,
                f"Impute missing values in {columns} using {strategy}",
            )

        if preview_btn and columns:
            params = {
                "columns": columns,
                "strategy": strategy,
                "fill_value": fill_value,
            }
            _preview_manual_action("handle_missing", params, dataset)

    _show_manual_preview()


# ══════════════════════════════════════════════════════════════
#  Tab 3: Duplicates
# ══════════════════════════════════════════════════════════════

def _render_duplicates(dataset):
    """Manual duplicate handling interface."""
    st.markdown("### 👥 Handle Duplicates")
    df = dataset.df
    dup_count = df.duplicated().sum()

    st.metric("Duplicate Rows", f"{dup_count:,}")

    if dup_count == 0:
        st.success("✅ No duplicate rows found!")
        return

    with st.expander("👁️ Preview Duplicate Rows"):
        dups = df[df.duplicated(keep=False)]
        st.dataframe(dups.head(50), width='stretch', height=300)

    with st.form("dup_form"):
        subset = render_column_selector(
            df, key="dup_cols", label="Subset columns (empty = all)"
        )
        keep = st.selectbox("Keep", ["first", "last", "none"])

        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button(
                "✅ Remove Duplicates", width='stretch')
        with col2:
            preview_btn = st.form_submit_button(
                "👁️ Preview", width='stretch')

        if submitted:
            params = {"subset": subset if subset else None, "keep": keep}
            _execute_user_action(
                "handle_duplicates", params, dataset,
                f"Remove duplicates keeping '{keep}'",
            )

        if preview_btn:
            params = {"subset": subset if subset else None, "keep": keep}
            _preview_manual_action("handle_duplicates", params, dataset)

    _show_manual_preview()


# ══════════════════════════════════════════════════════════════
#  Tab 4: Outliers
# ══════════════════════════════════════════════════════════════

def _render_outliers(dataset):
    """Manual outlier handling interface."""
    st.markdown("### 📊 Handle Outliers")
    df = dataset.df
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if not numeric_cols:
        st.info("No numeric columns available.")
        return

    with st.form("outlier_form"):
        columns = render_column_selector(
            df, key="outlier_cols", label="Select numeric columns",
            type_filter="numeric",
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            method = st.selectbox("Method", ["iqr", "zscore", "percentile"])
        with col2:
            threshold = st.number_input("Threshold", value=1.5, step=0.1)
        with col3:
            behavior = st.selectbox("Behavior", ["clip", "remove", "flag"])

        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button(
                "✅ Apply", width='stretch')
        with col2:
            preview_btn = st.form_submit_button(
                "👁️ Preview", width='stretch')

        if submitted and columns:
            params = {
                "columns": columns,
                "method": method,
                "threshold": threshold,
                "behavior": behavior,
            }
            _execute_user_action(
                "handle_outliers", params, dataset,
                f"Handle outliers in {columns} using {method} → {behavior}",
            )

        if preview_btn and columns:
            params = {
                "columns": columns,
                "method": method,
                "threshold": threshold,
                "behavior": behavior,
            }
            _preview_manual_action("handle_outliers", params, dataset)

    _show_manual_preview()


# ══════════════════════════════════════════════════════════════
#  Tab 5: Text Cleaning
# ══════════════════════════════════════════════════════════════

def _render_text_cleaning(dataset):
    """Manual text cleaning interface."""
    st.markdown("### ✏️ Text Cleaning")
    df = dataset.df
    text_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if not text_cols:
        st.info("No text columns available.")
        return

    with st.form("text_form"):
        columns = render_column_selector(
            df, key="text_cols", label="Select text columns",
            type_filter="text",
        )

        operations = st.multiselect(
            "Operations",
            ["trim_whitespace", "lowercase", "uppercase", "titlecase",
             "remove_special_chars", "remove_punctuation", "collapse_whitespace",
             "strip_html"],
        )

        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button(
                "✅ Apply", width='stretch')
        with col2:
            preview_btn = st.form_submit_button(
                "👁️ Preview", width='stretch')

        if submitted and columns and operations:
            params = {"columns": columns, "operations": operations}
            _execute_user_action(
                "text_cleaning", params, dataset,
                f"Text cleaning [{', '.join(operations)}] on {columns}",
            )

        if preview_btn and columns and operations:
            params = {"columns": columns, "operations": operations}
            _preview_manual_action("text_cleaning", params, dataset)

    _show_manual_preview()


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

        # ── Track this action to prevent re-recommendation ──
        sig = _action_signature(recommendation)
        applied = st.session_state.get("clean_applied_actions", set())
        applied.add(sig)
        st.session_state["clean_applied_actions"] = applied

        # ── Remove ONLY this action from cache ───────────────
        # This is the key fix: we do NOT nuke the entire cache.
        # AI recs and other static recs stay intact.
        recs = st.session_state.get("clean_recs_cache", [])
        st.session_state["clean_recs_cache"] = [
            r for r in recs
            if r.get("action_id") != recommendation.get("action_id")
        ]

        # ── Store success feedback ────────────────────────────
        desc = recommendation.get("description", "Action")
        rows_before = result.get("rows_before", "?")
        rows_after = result.get("rows_after", "?")
        st.session_state["clean_last_applied"] = (
            f"{desc}  (rows: {rows_before} → {rows_after})"
        )

        # ── Clear preview only ────────────────────────────────
        st.session_state.pop("clean_rec_preview_data", None)
        st.session_state.pop("manual_preview_data", None)

        StateManager.add_notification(
            f"✅ Applied: {desc}", "success"
        )
        st.rerun()
    else:
        error_msg = result.get("error", "Unknown error")
        st.session_state["clean_last_error"] = error_msg
        pipeline.remove_step(step.action_id)
        st.rerun()


def _reject_action(recommendation: dict):
    """Dismiss a recommendation.
    Removes ONLY this action from cache — preserves all others.
    """
    recs = st.session_state.get("clean_recs_cache", [])
    st.session_state["clean_recs_cache"] = [
        r for r in recs
        if r.get("action_id") != recommendation.get("action_id")
    ]
    st.session_state.pop("clean_rec_preview_data", None)
    st.rerun()


def _preview_action(recommendation: dict, dataset, key_prefix: str):
    """Preview a recommended action — stores keyed by action_id for inline rendering."""
    action_id = recommendation.get("action_id", "unknown")
    step = PipelineStep.from_dict(recommendation)
    pipeline = StateManager.get_pipeline()
    preview = pipeline.preview_step(step, dataset)
    # Store under action_id key so action_list renders it inline
    st.session_state[f"{key_prefix}_preview_{action_id}"] = preview
def _preview_manual_action(action_type: str, params: dict, dataset):
    """Preview a manually configured action."""
    action_class = ActionRegistry.get(action_type)
    if not action_class:
        st.error(f"Unknown action type: {action_type}")
        return

    try:
        action = action_class()
        errors = action.validate(dataset.df, params)
        if errors:
            st.error(f"Validation errors: {errors}")
            return

        preview = action.preview(dataset.df, params)
        st.session_state["manual_preview_data"] = preview
    except Exception as e:
        st.error(f"Preview failed: {e}")


def _show_manual_preview():
    """Show stored manual preview data below the form."""
    preview_data = st.session_state.get("manual_preview_data")
    if preview_data is not None:
        render_preview_table(preview_data)
        if st.button("🔒 Close Preview", key="close_manual_preview", width='stretch'):
            st.session_state.pop("manual_preview_data", None)
            st.rerun()


def _execute_user_action(action_type: str, params: dict, dataset, description: str):
    """Execute a manually configured action from the manual tabs.
    Marks recs as stale but does NOT delete the cache — user clicks
    Generate Recommendations when ready for a fresh analysis.
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
        applied = st.session_state.get("clean_applied_actions", set())
        applied.add(sig)
        st.session_state["clean_applied_actions"] = applied

        # Mark recommendations as potentially stale (data changed)
        # but do NOT delete them — user can still approve remaining ones
        st.session_state["clean_recs_stale"] = True

        st.session_state.pop("clean_rec_preview_data", None)
        st.session_state.pop("manual_preview_data", None)

        st.session_state["clean_last_applied"] = description
        StateManager.add_notification(f"✅ {description}", "success")
        st.rerun()
    else:
        pipeline.remove_step(step.action_id)
        st.error(f"❌ Failed: {result.get('error')}")


main()