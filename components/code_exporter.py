"""
Display reproducible Python code for actions.
"""

import streamlit as st
from typing import List, Dict

from config.registry import ActionRegistry
from core.pipeline import Pipeline


def render_code_export(pipeline: Pipeline):
    """Render reproducible Python code for the entire pipeline."""
    if pipeline.step_count == 0:
        st.info("No pipeline steps to export as code.")
        return

    code_lines = [
        "import pandas as pd",
        "import numpy as np",
        "",
        "# Load data",
        "df = pd.read_csv('your_data.csv')  # Update with your file path",
        "",
        "# ── Pipeline Steps ──────────────────────────────────",
    ]

    for i, step in enumerate(pipeline.enabled_steps):
        action_class = ActionRegistry.get(step.action_type)
        if action_class:
            action = action_class()
            snippet = action.get_code_snippet(step.parameters)
            code_lines.append(f"\n# Step {i + 1}: {step.description}")
            code_lines.append(snippet)
        else:
            code_lines.append(f"\n# Step {i + 1}: {step.description}")
            code_lines.append(
                f"# Action type '{step.action_type}' — manual implementation needed")

    code_lines.extend([
        "",
        "# ── Export Result ────────────────────────────────────",
        "df.to_csv('cleaned_data.csv', index=False)",
        "print(f'Final shape: {df.shape}')",
    ])

    full_code = "\n".join(code_lines)

    st.code(full_code, language="python")

    st.download_button(
        label="📥 Download Python Script",
        data=full_code,
        file_name="dataprep_pipeline.py",
        mime="text/x-python",
        width='stretch',
    )
