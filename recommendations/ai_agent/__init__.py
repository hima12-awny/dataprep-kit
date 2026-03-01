"""
AI Agent module — Phase 2 implementation.
"""

from recommendations.ai_agent.base_agent import BaseAgent
from recommendations.ai_agent.description_agent import DescriptionAgent
from recommendations.ai_agent.action_agent import ActionRecommendationAgent
from recommendations.ai_agent.deduplication import deduplicate_recommendations
from recommendations.ai_agent.models import (
    DatasetDescription,
    CleaningRecommendationList,
    ConversionRecommendationList,
    EngineeringRecommendationList,
)