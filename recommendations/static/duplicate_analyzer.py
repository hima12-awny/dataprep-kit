"""
Analyzes the dataframe for duplicate rows and recommends removal.
"""

import pandas as pd
from typing import Dict, List

from recommendations.static.base_analyzer import BaseAnalyzer
from config.settings import settings


class DuplicateAnalyzer(BaseAnalyzer):
    name = "duplicate_analyzer"
    domain = "cleaning"

    def analyze(self, df: pd.DataFrame) -> List[Dict]:
        recommendations = []
        total_rows = len(df)

        if total_rows == 0:
            return recommendations

        dup_count = int(df.duplicated().sum())
        dup_pct = dup_count / total_rows

        if dup_count == 0:
            return recommendations

        if dup_pct > 0.10:
            priority = "high"
        elif dup_pct > settings.DUPLICATE_WARN_THRESHOLD:
            priority = "medium"
        else:
            priority = "low"

        recommendations.append(
            self._build_recommendation(
                action_type="handle_duplicates",
                description=(
                    f"Remove {dup_count} duplicate rows ({dup_pct:.1%} of data), "
                    f"keeping first occurrence"
                ),
                parameters={
                    "subset": None,  # All columns
                    "keep": "first",
                },
                priority=priority,
                reason=(
                    f"Found {dup_count} exact duplicate rows ({dup_pct:.1%}). "
                    f"Duplicates can bias model training and inflate metrics. "
                    f"Keeping the first occurrence is the safest default."
                ),
            )
        )

        return recommendations
