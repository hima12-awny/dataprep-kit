"""
Data Description Agent — analyzes the dataset and generates a structured description.
"""

from typing import Optional

import pandas as pd
from pydantic_ai import Agent

from recommendations.ai_agent.base_agent import BaseAgent
from recommendations.ai_agent.models import DatasetDescription
from recommendations.ai_agent.context_builder import ContextBuilder


DESCRIPTION_SYSTEM_PROMPT = f"""You are a senior data analyst expert. Your job is to analyze a dataset
and produce a structured description that will help other AI agents and data engineers understand
what this data is about.

Analyze the column names, data types, sample values, and statistics provided.
Infer the domain, what each row represents, and assess data quality.

Be specific and practical. Mention actual column names and observed patterns.
If you see potential issues (mixed types, suspicious values, encoding problems), call them out.
"""


class DescriptionAgent(BaseAgent):
    """Generates a structured DatasetDescription from raw data."""

    agent_name = "description_agent"
    agent_description = "Analyzes dataset and generates structured description"

    def generate_description(self, df: pd.DataFrame) -> DatasetDescription:
        """
        Analyze the dataframe and return a structured DatasetDescription.

        Args:
            df: The dataframe to analyze.

        Returns:
            DatasetDescription pydantic model.

        Raises:
            RuntimeError: If the agent is not configured or call fails.
        """
        if not self.is_configured():
            raise RuntimeError(
                "AI agent not configured. Please set API key in AI Settings.")

        self._set_api_key_env()

        context = ContextBuilder.build_description_context(df)

        agent = Agent(
            self.get_model_string(),
            output_type=DatasetDescription,
            system_prompt=DESCRIPTION_SYSTEM_PROMPT,
        )

        result = agent.run_sync(
            f"Analyze this dataset and generate a structured description:\n\n{context}\n\nYour output must follow the exact structured schema provided. this JSON schema:\n{DatasetDescription.model_json_schema()}"
        )

        return result.output
