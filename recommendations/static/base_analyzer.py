"""
BaseAnalyzer — abstract class for all static recommendation analyzers.
"""

from abc import ABC, abstractmethod
from typing import Dict, List
from datetime import datetime, timezone

import pandas as pd

from utils.id_generator import generate_action_id


class BaseAnalyzer(ABC):
    """
    Base class for static rule-based analyzers.
    Each analyzer inspects the dataframe and produces a list of
    recommended Action JSON dicts.
    """

    name: str = "base_analyzer"
    domain: str = "cleaning"  # "cleaning", "conversion", "feature_engineering"

    @abstractmethod
    def analyze(self, df: pd.DataFrame) -> List[Dict]:
        """
        Analyze the dataframe and return a list of recommended actions.

        Each recommendation is a standard Action JSON dict:
        {
            "action_id": "...",
            "action_type": "...",
            "description": "...",
            "author": "ai_static",
            "timestamp": "...",
            "parameters": {...},
            "preview_only": False,
            "priority": "high" | "medium" | "low",
            "reason": "Why this action is recommended",
        }
        """
        pass

    def _build_recommendation(
        self,
        action_type: str,
        description: str,
        parameters: Dict,
        priority: str = "medium",
        reason: str = "",
    ) -> Dict:
        """Helper to construct a well-formed recommendation dict."""
        return {
            "action_id": generate_action_id("rec"),
            "action_type": action_type,
            "description": description,
            "author": "ai_static",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "parameters": parameters,
            "preview_only": False,
            "enabled": True,
            "priority": priority,
            "reason": reason,
        }
