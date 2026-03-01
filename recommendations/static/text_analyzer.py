"""
Analyzes text columns for whitespace issues, mixed case, and other problems.
"""

import pandas as pd
from typing import Dict, List

from recommendations.static.base_analyzer import BaseAnalyzer


class TextAnalyzer(BaseAnalyzer):
    name = "text_analyzer"
    domain = "cleaning"

    def analyze(self, df: pd.DataFrame) -> List[Dict]:
        recommendations = []
        text_cols = df.select_dtypes(include=["object"]).columns.tolist()

        for col in text_cols:
            non_null = df[col].dropna()
            if len(non_null) == 0:
                continue

            str_values = non_null.astype(str)
            operations_needed = []
            reasons = []

            # ── Whitespace issues ─────────────────────────────
            has_leading_trailing = (str_values != str_values.str.strip()).any()
            if has_leading_trailing:
                ws_count = int((str_values != str_values.str.strip()).sum())
                operations_needed.append("trim_whitespace")
                reasons.append(
                    f"{ws_count} values have leading/trailing whitespace"
                )

            # ── Extra internal whitespace ─────────────────────
            has_extra_spaces = str_values.str.contains(
                r"\s{2,}", regex=True, na=False).any()
            if has_extra_spaces:
                sp_count = int(str_values.str.contains(
                    r"\s{2,}", regex=True, na=False).sum())
                operations_needed.append("collapse_whitespace")
                reasons.append(f"{sp_count} values have extra internal spaces")

            # ── Mixed case (for low-cardinality columns) ──────
            n_unique = str_values.nunique()
            n_unique_lower = str_values.str.lower().nunique()
            if n_unique_lower < n_unique and n_unique <= 100:
                diff = n_unique - n_unique_lower
                operations_needed.append("lowercase")
                reasons.append(
                    f"{diff} duplicate categories caused by case differences "
                    f"(e.g., unique values drop from {n_unique} to {n_unique_lower} "
                    f"after lowercasing)"
                )

            if not operations_needed:
                continue

            priority = "high" if "lowercase" in operations_needed and len(
                reasons) > 1 else "medium"

            recommendations.append(
                self._build_recommendation(
                    action_type="text_cleaning",
                    description=(
                        f"Clean text in '{col}': {', '.join(operations_needed)}"
                    ),
                    parameters={
                        "columns": [col],
                        "operations": operations_needed,
                    },
                    priority=priority,
                    reason=f"Column '{col}': {'; '.join(reasons)}.",
                )
            )

        return recommendations
