"""
AI Settings page — configure provider, API key, data description, target tracks.
Dataset description editor uses proper UI widgets instead of raw JSON.
"""

import streamlit as st
import json
import pandas as pd
from typing import Dict, Optional

from core.state import StateManager
from config.settings import settings


# ══════════════════════════════════════════════════════════════
#  Helper Functions (must be defined BEFORE page-level code)
# ══════════════════════════════════════════════════════════════

def _render_description_editor(generated: Dict, dataset):
    """Render the full structured description editor with proper UI widgets."""

    df_columns = dataset.df.columns.tolist() if dataset else []

    # ── 1. Summary ────────────────────────────────────────
    st.markdown("##### 📝 Summary")
    new_summary = st.text_area(
        "Dataset Summary",
        value=generated.get("summary", ""),
        height=200,
        key="desc_edit_summary",
        help="2-3 sentence summary of what this dataset contains",
    )

    # ── 2. Domain & Row Description ───────────────────────
    col1, col2 = st.columns(2)
    with col1:
        new_domain = st.text_input(
            "Domain / Industry",
            value=generated.get("domain", ""),
            key="desc_edit_domain",
            placeholder="e.g., healthcare, finance, e-commerce",
        )
    with col2:
        new_row_desc = st.text_input(
            "What Each Row Represents",
            value=generated.get("row_description", ""),
            key="desc_edit_row",
            placeholder="e.g., one customer transaction",
        )

    # ── 3. Column Descriptions ────────────────────────────
    st.markdown("##### 📊 Column Descriptions")
    st.caption("Edit role, quality, and description for each column.")

    col_descs = generated.get("column_descriptions", [])
    col_desc_map = {cd.get("name", ""): cd for cd in col_descs}

    role_options = ["feature", "target", "identifier", "metadata", "drop"]
    quality_options = ["good", "needs_attention", "poor"]

    edited_columns = []

    for i, col_name in enumerate(df_columns):
        existing = col_desc_map.get(col_name, {})

        with st.expander(f"📌 `{col_name}`", expanded=False):
            c1, c2 = st.columns(2)

            with c1:
                role_default = existing.get("suggested_role", "feature")
                role_idx = role_options.index(
                    role_default) if role_default in role_options else 0
                new_role = st.selectbox(
                    "Role",
                    options=role_options,
                    index=role_idx,
                    key=f"desc_col_role_{i}",
                )

            with c2:
                qual_default = existing.get("data_quality", "good")
                qual_idx = quality_options.index(
                    qual_default) if qual_default in quality_options else 0
                new_quality = st.selectbox(
                    "Data Quality",
                    options=quality_options,
                    index=qual_idx,
                    key=f"desc_col_quality_{i}",
                )

            new_col_desc = st.text_input(
                "Description",
                value=existing.get("description", ""),
                key=f"desc_col_desc_{i}",
                placeholder=f"What does '{col_name}' represent?",
            )

            new_quality_notes = st.text_area(
                "Quality Notes",
                value=existing.get("quality_notes", ""),
                key=f"desc_col_notes_{i}",
                placeholder="Any specific quality observations",
                height=250
            )

            edited_columns.append({
                "name": col_name,
                "description": new_col_desc,
                "suggested_role": new_role,
                "data_quality": new_quality,
                "quality_notes": new_quality_notes,
            })

    # ── 4. Data Quality Notes ─────────────────────────────
    st.markdown("##### 🔍 Data Quality Notes")
    st.caption("Overall observations about data quality. One per line.")

    existing_quality_notes = generated.get("data_quality_notes", [])
    quality_notes_text = st.text_area(
        "Quality Notes",
        value="\n".join(existing_quality_notes),
        height=120,
        key="desc_edit_quality_notes",
        placeholder="One observation per line\ne.g., Several columns have >20% missing values",
    )

    # ── 5. Potential Issues ───────────────────────────────
    st.markdown("##### ⚠️ Potential Issues")
    st.caption("Known problems or concerns. One per line.")

    existing_issues = generated.get("potential_issues", [])
    issues_text = st.text_area(
        "Potential Issues",
        value="\n".join(existing_issues),
        height=120,
        key="desc_edit_issues",
        placeholder="One issue per line\ne.g., 'status' column has inconsistent casing",
    )

    # ── 6. Recommended Target Columns ─────────────────────
    st.markdown("##### 🎯 Recommended Target Columns")
    st.caption("Which columns could serve as prediction targets?")

    existing_targets = generated.get("recommended_target_columns", [])
    valid_targets = [t for t in existing_targets if t in df_columns]
    new_targets = st.multiselect(
        "Target Columns",
        options=df_columns,
        default=valid_targets,
        key="desc_edit_targets",
    )

    # ── Action Buttons ────────────────────────────────────
    st.markdown("---")
    col_approve, col_discard = st.columns(2)

    with col_approve:
        if st.button(
            "✅ Approve & Save Description",
            key="approve_ai_desc",
            width='stretch',
            type="primary",
        ):
            quality_notes_list = [
                line.strip() for line in quality_notes_text.split("\n")
                if line.strip()
            ]
            issues_list = [
                line.strip() for line in issues_text.split("\n")
                if line.strip()
            ]

            final_desc = {
                "summary": new_summary,
                "domain": new_domain,
                "row_description": new_row_desc,
                "column_descriptions": edited_columns,
                "data_quality_notes": quality_notes_list,
                "potential_issues": issues_list,
                "recommended_target_columns": new_targets,
                "source": "ai_edited",
            }

            StateManager.update_ai_config(
                data_description=final_desc,
                data_description_text=new_summary,
            )
            st.session_state.pop("ai_generated_description", None)
            st.success("✅ Description approved and saved!")

    with col_discard:
        if st.button("❌ Discard", key="discard_ai_desc", width='stretch'):
            st.session_state.pop("ai_generated_description", None)
            st.rerun()


# ══════════════════════════════════════════════════════════════
#  PAGE STARTS HERE
# ══════════════════════════════════════════════════════════════

st.markdown("## 🤖 AI Settings")
st.caption("Configure the AI agent for smart, context-aware recommendations")

st.markdown("---")

StateManager.initialize()
ai_config = StateManager.get_ai_config()

# ══════════════════════════════════════════════════════════════
#  Section 1: API Configuration
# ══════════════════════════════════════════════════════════════

st.markdown("### 1. API Configuration")
st.markdown("_The AI agent is optional. All static features work without it._")

col1, col2 = st.columns(2)

with col1:
    provider = st.selectbox(
        "Provider",
        options=list(settings.AI_MODELS.keys()),
        index=list(settings.AI_MODELS.keys()).index(
            ai_config.get("provider", "openai")),
        key="ai_provider",
    )

with col2:
    models = settings.AI_MODELS.get(provider, ["gpt-4o-mini"])
    current_model = ai_config.get("model", models[0])
    if current_model not in models:
        current_model = models[0]
    model = st.selectbox(
        "Model",
        options=models,
        index=models.index(current_model),
        key="ai_model",
    )

api_key = st.text_input(
    "API Key",
    value=ai_config.get("api_key", ""),
    type="password",
    placeholder="sk-... or your provider's API key",
    key="ai_api_key",
)

if st.button("💾 Save API Configuration", width='stretch'):
    StateManager.update_ai_config(
        provider=provider,
        model=model,
        api_key=api_key,
    )
    if api_key:
        st.success("✅ AI agent configured successfully!")
    else:
        st.info("API key cleared. AI features disabled.")

if ai_config.get("api_key"):
    st.success(
        f"✅ Connected: **{ai_config.get('provider')}** / **{ai_config.get('model')}**")
else:
    st.info("🔑 Enter an API key to enable AI-powered recommendations.")

st.markdown("---")

# ══════════════════════════════════════════════════════════════
#  Section 2: Target Use Cases
# ══════════════════════════════════════════════════════════════

st.markdown("### 2. Target Use Cases")
st.markdown(
    "_What are you preparing this data for? This helps the AI tailor its recommendations._")

target_tracks = st.multiselect(
    "Select one or more target tracks",
    options=list(settings.TARGET_TRACKS),
    default=ai_config.get("target_tracks", []),
    key="ai_target_tracks",
)

if st.button("💾 Save Target Tracks", width='stretch'):
    StateManager.update_ai_config(target_tracks=target_tracks)
    st.success("✅ Target tracks saved!")

st.markdown("---")

# ══════════════════════════════════════════════════════════════
#  Section 3: Data Description
# ══════════════════════════════════════════════════════════════

st.markdown("### 3. Data Description")
st.markdown(
    "_Provide a description of your dataset. This context helps the AI make better recommendations. "
    "You can write it yourself or let the AI generate it._"
)

dataset = StateManager.get_dataset() if StateManager.has_dataset() else None

tab_manual, tab_ai = st.tabs(
    ["✍️ Manual Description", "🤖 AI-Generated Description"])

# ── Tab 1: Manual Description ─────────────────────────────────
with tab_manual:
    current_text = ai_config.get("data_description_text", "")
    description_text = st.text_area(
        "Describe your dataset",
        value=current_text,
        height=200,
        placeholder=(
            "Example: This dataset contains customer transaction records from an e-commerce platform. "
            "Each row represents a single purchase. The 'revenue' column is the target for prediction. "
            "The data spans Jan 2022 to Dec 2023 and includes both numeric and categorical features..."
        ),
        key="manual_desc_text",
    )

    if st.button("💾 Save Description", key="save_manual_desc", width='stretch'):
        StateManager.update_ai_config(
            data_description_text=description_text,
            data_description={"summary": description_text, "source": "manual"},
        )
        st.success("✅ Description saved!")

# ── Tab 2: AI-Generated Description ──────────────────────────
with tab_ai:
    if not dataset:
        st.warning("⚠️ Import a dataset first to generate an AI description.")
    elif not ai_config.get("api_key"):
        st.warning("⚠️ Configure API key above first.")
    else:
        st.markdown(
            "The AI will analyze your data and generate a structured description. "
            "You can then **edit every field** before approving."
        )

        if st.button(
            "🤖 Generate Description with AI",
            key="gen_ai_desc",
            width='stretch',
            type="primary",
        ):
            with st.spinner("AI is analyzing your dataset..."):
                try:
                    from recommendations.ai_agent.description_agent import DescriptionAgent

                    agent = DescriptionAgent(
                        provider=ai_config.get("provider", "openai"),
                        model=ai_config.get("model", "gpt-4o-mini"),
                        api_key=ai_config.get("api_key", ""),
                    )

                    description = agent.generate_description(dataset.df)
                    st.session_state["ai_generated_description"] = description.model_dump(
                    )
                    st.success(
                        "✅ Description generated! Review and edit below, then approve.")

                except Exception as e:
                    st.error(f"❌ AI generation failed: {e}")

        # Determine what to show in the editor
        generated = st.session_state.get("ai_generated_description")

        # Also allow re-editing a previously saved structured description
        if not generated:
            saved_desc = ai_config.get("data_description")
            if (saved_desc
                    and isinstance(saved_desc, dict)
                    and saved_desc.get("source") != "manual"):
                generated = saved_desc
                st.info(
                    "📝 Editing your previously saved AI description. Make changes and re-approve.")

        if generated:
            st.markdown("---")
            st.markdown("#### 📋 Edit Dataset Description")
            st.caption(
                "Edit any field below using the UI controls, then click Approve to save.")
            _render_description_editor(generated, dataset)

st.markdown("---")

# ══════════════════════════════════════════════════════════════
#  Section 4: Current Configuration Summary
# ══════════════════════════════════════════════════════════════

st.markdown("### 📊 Current Configuration Summary")

summary_data = {
    "Provider": ai_config.get("provider", "Not set"),
    "Model": ai_config.get("model", "Not set"),
    "API Key": "✅ Set" if ai_config.get("api_key") else "❌ Not set",
    "Target Tracks": ", ".join(ai_config.get("target_tracks", [])) or "None selected",
    "Data Description": "✅ Provided" if ai_config.get("data_description") else "❌ Not provided",
}

for key, val in summary_data.items():
    st.markdown(f"**{key}:** {val}")
