"""
Analyzes columns and suggests feature engineering opportunities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List

from recommendations.static.base_analyzer import BaseAnalyzer
from utils.stats_helpers import StatsHelper
from config.settings import settings


class FeatureAnalyzer(BaseAnalyzer):
    name = "feature_analyzer"
    domain = "feature_engineering"

    def analyze(self, df: pd.DataFrame) -> List[Dict]:
        recommendations = []

        self._check_datetime_features(df, recommendations)
        self._check_high_correlation(df, recommendations)
        self._check_low_variance(df, recommendations)
        self._check_rare_categories(df, recommendations)

        return recommendations

    def _check_datetime_features(self, df: pd.DataFrame, recs: List[Dict]):
        """Suggest extracting components from datetime columns."""
        datetime_cols = df.select_dtypes(
            include=["datetime64"]).columns.tolist()

        for col in datetime_cols:
            recs.append(
                self._build_recommendation(
                    action_type="datetime_ops",
                    description=(
                        f"Extract date components (year, month, weekday, is_weekend) "
                        f"from '{col}'"
                    ),
                    parameters={
                        "operation": "extract_components",
                        "column": col,
                        "components": ["year", "month", "day", "weekday", "is_weekend"],
                        "prefix": col,
                    },
                    priority="medium",
                    reason=(
                        f"Datetime column '{col}' can provide useful features like "
                        f"year, month, day of week, and weekend flag for modeling."
                    ),
                )
            )

    def _check_high_correlation(self, df: pd.DataFrame, recs: List[Dict]):
        """Suggest dropping one of highly correlated feature pairs."""
        pairs = StatsHelper.high_correlation_pairs(
            df, threshold=settings.CORRELATION_HIGH_THRESHOLD
        )

        for col1, col2, corr_val in pairs[:5]:  # Limit to top 5 pairs
            recs.append(
                self._build_recommendation(
                    action_type="column_ops",
                    description=(
                        f"Consider dropping '{col2}' — highly correlated with "
                        f"'{col1}' (r={corr_val:.3f})"
                    ),
                    parameters={
                        "operation": "drop_columns",
                        "columns": [col2],
                    },
                    priority="low",
                    reason=(
                        f"Columns '{col1}' and '{col2}' have correlation {corr_val:.3f} "
                        f"(threshold: {settings.CORRELATION_HIGH_THRESHOLD}). "
                        f"Highly correlated features add redundancy and can harm "
                        f"some models. Consider dropping one."
                    ),
                )
            )

    def _check_low_variance(self, df: pd.DataFrame, recs: List[Dict]):
        """Suggest dropping near-zero variance columns."""
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        for col in numeric_cols:
            variance = df[col].var()
            if variance is not None and variance < settings.VARIANCE_LOW_THRESHOLD:
                recs.append(
                    self._build_recommendation(
                        action_type="column_ops",
                        description=(
                            f"Consider dropping '{col}' — near-zero variance ({variance:.6f})"
                        ),
                        parameters={
                            "operation": "drop_columns",
                            "columns": [col],
                        },
                        priority="low",
                        reason=(
                            f"Column '{col}' has variance {variance:.6f}, below the "
                            f"threshold {settings.VARIANCE_LOW_THRESHOLD}. Near-constant "
                            f"features provide little predictive value."
                        ),
                    )
                )

    def _check_rare_categories(self, df: pd.DataFrame, recs: List[Dict]):
        """Suggest merging rare categories."""
        cat_cols = df.select_dtypes(
            include=["object", "category"]).columns.tolist()

        for col in cat_cols:
            freq = df[col].value_counts(normalize=True)
            rare = freq[freq < settings.RARE_CATEGORY_THRESHOLD]

            if len(rare) >= 3:  # Only recommend if 3+ rare categories
                recs.append(
                    self._build_recommendation(
                        action_type="inconsistency",
                        description=(
                            f"Merge {len(rare)} rare categories in '{col}' into 'Other' "
                            f"(each < {settings.RARE_CATEGORY_THRESHOLD:.0%} frequency)"
                        ),
                        parameters={
                            "columns": [col],
                            "operation": "merge_rare_categories",
                            "threshold": settings.RARE_CATEGORY_THRESHOLD,
                            "replacement": "Other",
                        },
                        priority="low",
                        reason=(
                            f"Column '{col}' has {len(rare)} categories each representing "
                            f"less than {settings.RARE_CATEGORY_THRESHOLD:.0%} of the data. "
                            f"Rare categories can cause noise in encoding and model training."
                        ),
                    )
                )
