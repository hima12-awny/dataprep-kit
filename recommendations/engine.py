"""
RecommendationEngine: orchestrates static analyzers + AI agent.
Returns a unified, deduplicated list of action dicts.
"""

import pandas as pd
from typing import Dict, List, Optional

from core.dataset import Dataset


class RecommendationEngine:
    """
    Central engine that collects recommendations from static analyzers
    and optionally from the AI agent, then deduplicates them.
    """

    def __init__(self):
        self._static_analyzers = []
        self._load_static_analyzers()

    def _load_static_analyzers(self):
        from recommendations.static.missing_analyzer import MissingAnalyzer
        from recommendations.static.type_analyzer import TypeAnalyzer
        from recommendations.static.outlier_analyzer import OutlierAnalyzer
        from recommendations.static.duplicate_analyzer import DuplicateAnalyzer
        from recommendations.static.text_analyzer import TextAnalyzer
        from recommendations.static.feature_analyzer import FeatureAnalyzer

        self._static_analyzers = [
            MissingAnalyzer(),
            TypeAnalyzer(),
            OutlierAnalyzer(),
            DuplicateAnalyzer(),
            TextAnalyzer(),
            FeatureAnalyzer(),
        ]

    def get_recommendations(
        self,
        dataset: Dataset,
        domain: Optional[str] = None,
        ai_config: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Get all recommendations for the current dataset.

        Args:
            dataset: Current Dataset object.
            domain: Filter by domain ('cleaning', 'conversion', 'feature_engineering').
            ai_config: AI configuration dict with provider, model, api_key, etc.
                       If None or no api_key, only static recommendations are returned.
        """
        # ── Static recommendations ────────────────────────────
        static_recs = self._get_static_recommendations(dataset.df, domain)

        # ── AI recommendations (if configured) ────────────────
        ai_recs = []
        if ai_config and ai_config.get("api_key") and domain:
            ai_recs = self._get_ai_recommendations(dataset.df, domain, ai_config)

        # ── Deduplicate and merge ─────────────────────────────
        if ai_recs:
            from recommendations.ai_agent.deduplication import deduplicate_recommendations
            all_recs = deduplicate_recommendations(static_recs, ai_recs)
        else:
            all_recs = static_recs

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        all_recs.sort(key=lambda r: priority_order.get(r.get("priority", "low"), 3))

        return all_recs

    def _get_static_recommendations(
        self, df: pd.DataFrame, domain: Optional[str]
    ) -> List[Dict]:
        static_recs = []
        for analyzer in self._static_analyzers:
            if domain and analyzer.domain != domain:
                continue
            try:
                recs = analyzer.analyze(df)
                static_recs.extend(recs)
            except Exception as e:
                static_recs.append({
                    "action_type": "error",
                    "description": f"Analyzer '{analyzer.name}' failed: {str(e)}",
                    "author": "system",
                    "priority": "low",
                    "parameters": {},
                })
        return static_recs

    def _get_ai_recommendations(
        self, df: pd.DataFrame, domain: str, ai_config: Dict
    ) -> List[Dict]:
        try:
            from recommendations.ai_agent.action_agent import ActionRecommendationAgent

            agent = ActionRecommendationAgent(
                provider=ai_config.get("provider", "groq"),
                model=ai_config.get("model", "llama-3.3-70b-versatile"),
                api_key=ai_config.get("api_key", ""),
            )

            return agent.get_recommendations(
                df=df,
                domain=domain,
                data_description=ai_config.get("data_description"),
                target_tracks=ai_config.get("target_tracks"),
            )
        except Exception as e:
            return [{
                "action_id": "ai_error",
                "action_type": "error",
                "description": f"AI Agent failed: {str(e)}",
                "author": "ai_agent",
                "priority": "low",
                "parameters": {},
                "reason": str(e),
            }]

    # ── Shortcut methods ──────────────────────────────────────

    def get_cleaning_recommendations(
        self, dataset: Dataset, ai_config: Optional[Dict] = None
    ) -> List[Dict]:
        return self.get_recommendations(dataset, domain="cleaning", ai_config=ai_config)

    def get_conversion_recommendations(
        self, dataset: Dataset, ai_config: Optional[Dict] = None
    ) -> List[Dict]:
        return self.get_recommendations(dataset, domain="conversion", ai_config=ai_config)

    def get_engineering_recommendations(
        self, dataset: Dataset, ai_config: Optional[Dict] = None
    ) -> List[Dict]:
        return self.get_recommendations(dataset, domain="feature_engineering", ai_config=ai_config)