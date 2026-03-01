"""
Domain Action Recommendation Agent — generates domain-specific action recommendations.
One call per domain, using the domain's focused toolset and output schema.
"""

from typing import Dict, List, Optional
import pandas as pd
from pydantic_ai import Agent

from recommendations.ai_agent.base_agent import BaseAgent
from recommendations.ai_agent.models import (
    CleaningRecommendationList,
    ConversionRecommendationList,
    EngineeringRecommendationList,
)
from recommendations.ai_agent.context_builder import ContextBuilder

from recommendations.ai_agent.toolsets.cleaning_toolset import CLEANING_TOOLSET_PROMPT
from recommendations.ai_agent.toolsets.conversion_toolset import CONVERSION_TOOLSET_PROMPT
from recommendations.ai_agent.toolsets.engineering_toolset import ENGINEERING_TOOLSET_PROMPT

from recommendations.ai_agent.prompts.system_prompt import BASE_SYSTEM_PROMPT
from recommendations.ai_agent.prompts.cleaning_prompt import CLEANING_DOMAIN_PROMPT
from recommendations.ai_agent.prompts.conversion_prompt import CONVERSION_DOMAIN_PROMPT
from recommendations.ai_agent.prompts.engineering_prompt import ENGINEERING_DOMAIN_PROMPT
from utils.id_generator import generate_action_id
from rich import print as rp


# Map domain → (output model, toolset prompt, domain prompt)
DOMAIN_CONFIG = {
    "cleaning": {
        "model": CleaningRecommendationList,
        "toolset_prompt": CLEANING_TOOLSET_PROMPT,
        "domain_prompt": CLEANING_DOMAIN_PROMPT,
    },
    "conversion": {
        "model": ConversionRecommendationList,
        "toolset_prompt": CONVERSION_TOOLSET_PROMPT,
        "domain_prompt": CONVERSION_DOMAIN_PROMPT,
    },
    "feature_engineering": {
        "model": EngineeringRecommendationList,
        "toolset_prompt": ENGINEERING_TOOLSET_PROMPT,
        "domain_prompt": ENGINEERING_DOMAIN_PROMPT,
    },
}


class ActionRecommendationAgent(BaseAgent):
    """
    Generates domain-specific action recommendations.
    Each domain call sends:
    1. Base system prompt
    2. Domain-specific system prompt
    3. Toolset schema (injected in prompt)
    4. Dataset context (stats + description + target tracks)
    """

    agent_name = "action_agent"
    agent_description = "Recommends data preparation actions per domain"

    def get_recommendations(
        self,
        df: pd.DataFrame,
        domain: str,
        data_description: Optional[Dict] = None,
        target_tracks: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Get AI recommendations for a specific domain.

        Args:
            df: Current dataframe.
            domain: One of 'cleaning', 'conversion', 'feature_engineering'.
            data_description: Structured data description dict.
            target_tracks: List of target use cases.

        Returns:
            List of action dicts in the standard pipeline format.
        """

        if not self.is_configured():
            raise RuntimeError(
                "AI agent not configured. Please set API key in AI Settings.")

        if domain not in DOMAIN_CONFIG:
            raise ValueError(f"Unknown domain: {domain}")

        self._set_api_key_env()

        config = DOMAIN_CONFIG[domain]

        # Build the combined system prompt
        system_prompt = self._build_system_prompt(
            config["domain_prompt"],
            config["toolset_prompt"],
            config["model"],
        )

        # Build the dataset context
        context = ContextBuilder.build_full_context(
            df,
            data_description=data_description,
            target_tracks=target_tracks,
        )

        # Create the agent with the domain-specific output model
        agent = Agent(
            self.get_model_string(),
            output_type=config["model"],
            system_prompt=system_prompt,
            model_settings={
                "temperature": 0.5,
            },
        )

        # Run the agent
        result = agent.run_sync(
            f"Analyze this dataset and recommend {domain} actions:\n\n{context}"
        )

        # Convert pydantic models to standard action dicts
        return self._to_action_dicts(result.output, domain)

    def _build_system_prompt(self, domain_prompt: str, toolset_prompt: str, model) -> str:
        """Combine base + domain + toolset into a single system prompt."""
        return f"{BASE_SYSTEM_PROMPT}\n\n{domain_prompt}\n\n{toolset_prompt}\n\n and make sure you output formste match with this schema {model.model_json_schema()}"

    def _to_action_dicts(self, recommendation_list, domain: str) -> List[Dict]:
        """Convert a pydantic RecommendationList to standard action dicts."""
        from datetime import datetime, timezone

        actions = []
        for rec in recommendation_list.recommendations:
            # Convert parameters model to plain dict, dropping None values
            params = rec.parameters.model_dump(exclude_none=True)

            actions.append({
                "action_id": generate_action_id("ai"),
                "action_type": rec.action_type,
                "description": rec.description,
                "author": "ai_agent",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "parameters": params,
                "preview_only": False,
                "enabled": True,
                "priority": rec.priority,
                "reason": rec.reason,
            })

        rp(actions)
        return actions
