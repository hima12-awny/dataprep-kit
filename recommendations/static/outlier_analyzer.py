"""
Analyzes numeric columns for outliers and recommends handling strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List

from recommendations.static.base_analyzer import BaseAnalyzer
from utils.stats_helpers import StatsHelper
from config.settings import settings


class OutlierAnalyzer(BaseAnalyzer):
    name = "outlier_analyzer"
    domain = "cleaning"

    def analyze(self, df: pd.DataFrame) -> List[Dict]:
        recommendations = []
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        for col in numeric_cols:
            if df[col].isna().all():  # type: ignore
                continue

            outlier_info = StatsHelper.detect_outliers(
                df[col],  # type: ignore
                threshold=settings.IQR_MULTIPLIER
            )
            outlier_pct = outlier_info.get("outlier_percentage", 0)
            outlier_count = outlier_info.get("outlier_count", 0)

            if outlier_count == 0:
                continue

            # Decide behavior based on percentage
            if outlier_pct > 10:
                # Many outliers — clipping is safer than removal
                priority = "high"
                behavior = "clip"
                reason_extra = (
                    "High outlier percentage — clipping (Winsorizing) recommended "
                    "to avoid losing too many rows."
                )
            elif outlier_pct > 3:
                priority = "medium"
                behavior = "clip"
                reason_extra = (
                    "Moderate outlier presence — clipping preserves data volume."
                )
            elif outlier_pct > 0.5:
                priority = "low"
                behavior = "flag"
                reason_extra = (
                    "Small number of outliers — flagging recommended so you can "
                    "decide case-by-case."
                )
            else:
                continue  # Negligible

            recommendations.append(
                self._build_recommendation(
                    action_type="handle_outliers",
                    description=(
                        f"Handle {outlier_count} outliers in '{col}' "
                        f"({outlier_pct:.1f}%) using IQR method → {behavior}"
                    ),
                    parameters={
                        "columns": [col],
                        "method": "iqr",
                        "threshold": settings.IQR_MULTIPLIER,
                        "behavior": behavior,
                    },
                    priority=priority,
                    reason=(
                        f"Column '{col}' has {outlier_count} outliers ({outlier_pct:.1f}%) "
                        f"based on IQR method (bounds: "
                        f"[{outlier_info.get('lower_bound')}, {outlier_info.get('upper_bound')}]). "
                        f"{reason_extra}"
                    ),
                )
            )

        return recommendations
