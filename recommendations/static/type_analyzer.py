"""
Analyzes column types and recommends type conversions using regex detection.
"""

import pandas as pd
from typing import Dict, List

from recommendations.static.base_analyzer import BaseAnalyzer
from utils.type_detector import TypeDetector


class TypeAnalyzer(BaseAnalyzer):
    name = "type_analyzer"
    domain = "conversion"

    def analyze(self, df: pd.DataFrame) -> List[Dict]:
        recommendations = []

        suggestions = TypeDetector.get_suggested_conversions(df)

        for suggestion in suggestions:
            col = suggestion["column"]
            current = suggestion["current_type"]
            suggested = suggestion["suggested_type"]
            confidence = suggestion["confidence"]
            reason = suggestion["reason"]

            if suggested == current:
                continue

            # Determine priority based on confidence
            if confidence >= 0.95:
                priority = "high"
            elif confidence >= 0.85:
                priority = "medium"
            else:
                priority = "low"

            recommendations.append(
                self._build_recommendation(
                    action_type="type_casting",
                    description=(
                        f"Convert '{col}' from {current} to {suggested} "
                        f"(confidence: {confidence:.0%})"
                    ),
                    parameters={
                        "conversions": {col: suggested},
                        "errors": "coerce",
                    },
                    priority=priority,
                    reason=(
                        f"Column '{col}' is currently '{current}' but appears to be "
                        f"'{suggested}'. {reason}."
                    ),
                )
            )

        return recommendations
