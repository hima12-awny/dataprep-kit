"""
Builds dataset context payload for the AI agents.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from utils.stats_helpers import StatsHelper


class ContextBuilder:
    """
    Constructs a compact but informative context string from the
    dataframe for the LLM to reason about.
    """

    @staticmethod
    def build_full_context(
        df: pd.DataFrame,
        data_description: Optional[Dict] = None,
        target_tracks: Optional[List[str]] = None,
        max_head_rows: int = 8,
    ) -> str:
        """
        Build a comprehensive context string combining data stats and user-provided info.
        This is sent to every domain agent.
        """
        parts = []

        # ── Shape & types ─────────────────────────────────────
        parts.append(f"## Dataset Shape\nRows: {len(df)}, Columns: {len(df.columns)}")

        # ── Column types ──────────────────────────────────────
        type_lines = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            null_count = int(df[col].isna().sum())
            null_pct = f"{df[col].isna().mean() * 100:.1f}%"
            unique = df[col].nunique()
            type_lines.append(
                f"  - {col}: {dtype} | nulls: {null_count} ({null_pct}) | unique: {unique}"
            )
        parts.append("## Column Info\n" + "\n".join(type_lines))

        # ── Data head ─────────────────────────────────────────
        head_str = df.head(max_head_rows).to_string(max_cols=20)
        parts.append(f"## Data Sample (first {max_head_rows} rows)\n```\n{head_str}\n```")

        # ── Statistics ────────────────────────────────────────
        try:
            desc = df.describe(include="all").round(3).to_string()
            parts.append(f"## Statistical Summary\n```\n{desc}\n```")
        except Exception:
            pass

        # ── Missing values summary ────────────────────────────
        missing = df.isna().sum()
        missing = missing[missing > 0]
        if len(missing) > 0:
            missing_lines = [f"  - {col}: {count} ({count/len(df)*100:.1f}%)" for col, count in missing.items()]
            parts.append("## Missing Values\n" + "\n".join(missing_lines))
        else:
            parts.append("## Missing Values\nNone detected.")

        # ── Duplicate info ────────────────────────────────────
        dup_count = int(df.duplicated().sum())
        parts.append(f"## Duplicates\n{dup_count} duplicate rows ({dup_count/max(len(df),1)*100:.1f}%)")

        # ── User-provided data description ────────────────────
        if data_description:
            parts.append("## Data Description (provided by user/AI)")
            if isinstance(data_description, dict):
                if data_description.get("summary"):
                    parts.append(f"Summary: {data_description['summary']}")
                if data_description.get("domain"):
                    parts.append(f"Domain: {data_description['domain']}")
                if data_description.get("row_description"):
                    parts.append(f"Each row represents: {data_description['row_description']}")
                if data_description.get("potential_issues"):
                    issues = ", ".join(data_description["potential_issues"])
                    parts.append(f"Known issues: {issues}")
            elif isinstance(data_description, str):
                parts.append(data_description)

        # ── Target tracks ─────────────────────────────────────
        if target_tracks:
            parts.append(f"## Target Use Cases\n{', '.join(target_tracks)}")
            parts.append(
                "Consider these use cases when recommending actions. "
                "For example, ML targets may need different encoding than EDA."
            )

        return "\n\n".join(parts)

    @staticmethod
    def build_description_context(df: pd.DataFrame, max_head_rows: int = 10) -> str:
        """
        Build a context specifically for the Data Description agent.
        Focuses on giving the LLM enough info to describe the dataset.
        """
        parts = []

        parts.append(f"Dataset has {len(df)} rows and {len(df.columns)} columns.")

        # Column details
        col_details = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            null_pct = f"{df[col].isna().mean() * 100:.1f}%"
            unique = df[col].nunique()
            sample_vals = df[col].dropna().head(5).tolist()
            sample_str = ", ".join(str(v) for v in sample_vals)
            col_details.append(
                f"  - '{col}' ({dtype}): {unique} unique, {null_pct} missing. Samples: [{sample_str}]"
            )
        parts.append("Columns:\n" + "\n".join(col_details))

        # Head
        head_str = df.head(max_head_rows).to_string(max_cols=25)
        parts.append(f"First {max_head_rows} rows:\n```\n{head_str}\n```")

        # Stats
        try:
            desc = df.describe(include="all").round(3).to_string()
            parts.append(f"Statistics:\n```\n{desc}\n```")
        except Exception:
            pass

        return "\n\n".join(parts)