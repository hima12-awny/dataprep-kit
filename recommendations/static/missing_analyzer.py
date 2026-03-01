"""
Analyzes missing values and recommends imputation or removal strategies.
"""

import pandas as pd
from typing import Dict, List

from recommendations.static.base_analyzer import BaseAnalyzer
from config.settings import settings


class MissingAnalyzer(BaseAnalyzer):
    name = "missing_analyzer"
    domain = "cleaning"

    def analyze(self, df: pd.DataFrame) -> List[Dict]:
        recommendations = []
        total_rows = len(df)

        if total_rows == 0:
            return recommendations

        for col in df.columns:
            null_count = int(df[col].isna().sum()) #type: ignore
            if null_count == 0:
                continue

            null_pct = null_count / total_rows

            # ── High missing: suggest drop column ─────────────
            if null_pct > settings.MISSING_HIGH_THRESHOLD:
                recommendations.append(
                    self._build_recommendation(
                        action_type="handle_missing",
                        description=(
                            f"Drop column '{col}' — {null_pct:.1%} missing values "
                            f"({null_count}/{total_rows} rows)"
                        ),
                        parameters={
                            "columns": [col],
                            "strategy": "drop_columns",
                            "threshold": settings.MISSING_HIGH_THRESHOLD,
                        },
                        priority="high",
                        reason=(
                            f"Column '{col}' has {null_pct:.1%} missing values, "
                            f"exceeding the {settings.MISSING_HIGH_THRESHOLD:.0%} threshold. "
                            f"Imputation may introduce too much noise."
                        ),
                    )
                )

            # ── Medium missing: suggest imputation ────────────
            elif null_pct > settings.MISSING_MEDIUM_THRESHOLD:
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Check skewness to decide mean vs median
                    skew = abs(df[col].skew())
                    strategy = "median" if skew > 1 else "mean"
                    reason_detail = (
                        f"Skewness is {skew:.2f} — "
                        f"{'median recommended (skewed distribution)' if skew > 1 else 'mean recommended (roughly symmetric)'}"
                    )

                    recommendations.append(
                        self._build_recommendation(
                            action_type="handle_missing",
                            description=(
                                f"Impute '{col}' missing values ({null_pct:.1%}) "
                                f"using {strategy}"
                            ),
                            parameters={
                                "columns": [col],
                                "strategy": strategy,
                            },
                            priority="medium",
                            reason=(
                                f"Column '{col}' has {null_pct:.1%} missing values. "
                                f"{reason_detail}."
                            ),
                        )
                    )
                else:
                    # Categorical: suggest mode
                    recommendations.append(
                        self._build_recommendation(
                            action_type="handle_missing",
                            description=(
                                f"Impute '{col}' missing values ({null_pct:.1%}) "
                                f"using mode (most frequent value)"
                            ),
                            parameters={
                                "columns": [col],
                                "strategy": "mode",
                            },
                            priority="medium",
                            reason=(
                                f"Categorical column '{col}' has {null_pct:.1%} missing. "
                                f"Mode imputation preserves the most common category."
                            ),
                        )
                    )

            # ── Low missing: flag only ────────────────────────
            elif null_pct > settings.MISSING_LOW_THRESHOLD:
                if pd.api.types.is_numeric_dtype(df[col]):
                    recommendations.append(
                        self._build_recommendation(
                            action_type="handle_missing",
                            description=(
                                f"Impute '{col}' ({null_count} missing values) using median"
                            ),
                            parameters={
                                "columns": [col],
                                "strategy": "median",
                            },
                            priority="low",
                            reason=(
                                f"Column '{col}' has a small number of missing values "
                                f"({null_pct:.1%}). Median is a safe default."
                            ),
                        )
                    )
                else:
                    recommendations.append(
                        self._build_recommendation(
                            action_type="handle_missing",
                            description=(
                                f"Impute '{col}' ({null_count} missing values) using mode"
                            ),
                            parameters={
                                "columns": [col],
                                "strategy": "mode",
                            },
                            priority="low",
                            reason=(
                                f"Column '{col}' has a small number of missing values "
                                f"({null_pct:.1%}). Mode is appropriate for categorical data."
                            ),
                        )
                    )

        return recommendations
