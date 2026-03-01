"""
BaseAgent — Abstract base class for all AI agents in DataPrep Kit.

Handles provider configuration, API key management, and model string
construction for pydantic-ai Agent instances.

Subclassed by:
    - DescriptionAgent (description_agent.py)
    - ActionRecommendationAgent (action_agent.py)
"""

import os
import logging
from typing import Dict

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
#  Provider Configuration
# ══════════════════════════════════════════════════════════════
# Maps provider names → env var for API key + pydantic-ai model prefix

PROVIDER_CONFIG: Dict[str, Dict[str, str]] = {
    "groq": {
        "env_var": "GROQ_API_KEY",
        "model_prefix": "groq",
    },
    "openai": {
        "env_var": "OPENAI_API_KEY",
        "model_prefix": "openai",
    },
    "anthropic": {
        "env_var": "ANTHROPIC_API_KEY",
        "model_prefix": "anthropic",
    },
    "google": {
        "env_var": "GOOGLE_API_KEY",
        "model_prefix": "google-gla",
    },
}


class BaseAgent:
    """
    Abstract base for all AI agents in DataPrep Kit.

    Responsibilities:
        - Store and validate provider credentials (provider, model, api_key).
        - Inject the API key into the environment so pydantic-ai can
          discover it automatically.
        - Construct the ``"prefix:model"`` string that pydantic-ai's
          ``Agent()`` constructor expects.

    Subclasses must define:
        agent_name  (str) — unique id used for logging and tracking.
        agent_description (str) — human-readable purpose.

    Typical subclass pattern::

        class MyAgent(BaseAgent):
            agent_name = "my_agent"
            agent_description = "Does something smart"

            def run(self, df):
                if not self.is_configured():
                    raise RuntimeError("Not configured")
                self._set_api_key_env()
                agent = Agent(self.get_model_string(), output_type=..., ...)
                result = agent.run_sync(prompt)
                return result.output
    """

    # ── Subclass identifiers (override in children) ───────────
    agent_name: str = "base_agent"
    agent_description: str = "Base AI agent"

    # ══════════════════════════════════════════════════════════
    #  Construction
    # ══════════════════════════════════════════════════════════

    def __init__(
        self,
        provider: str = "groq",
        model: str = "llama-3.3-70b-versatile",
        api_key: str = "",
    ):
        """
        Args:
            provider: AI provider key — one of 'groq', 'openai',
                      'anthropic', 'google'.  Unknown providers are
                      accepted with a warning (fallback prefix = provider name).
            model:    Model identifier within the provider
                      (e.g. 'llama-3.3-70b-versatile', 'gpt-4o-mini').
            api_key:  Secret key for the provider's API.
        """
        self.provider: str = provider.lower().strip()
        self.model: str = model.strip()
        self.api_key: str = api_key.strip() if api_key else ""

        if self.provider not in PROVIDER_CONFIG:
            logger.warning(
                "[%s] Unknown provider '%s'. Supported: %s. "
                "Will attempt '%s:%s' as model string.",
                self.agent_name,
                self.provider,
                list(PROVIDER_CONFIG.keys()),
                self.provider,
                self.model,
            )

    # ══════════════════════════════════════════════════════════
    #  Public helpers
    # ══════════════════════════════════════════════════════════

    def is_configured(self) -> bool:
        """
        Check whether the agent can make API calls.

        Returns:
            ``True`` when both *api_key* and *model* are non-empty.
        """
        return bool(self.api_key and self.model)

    def get_model_string(self) -> str:
        """
        Build the model string expected by ``pydantic_ai.Agent()``.

        Format: ``"<prefix>:<model>"``

        Examples:
            - ``"groq:llama-3.3-70b-versatile"``
            - ``"openai:gpt-4o-mini"``
            - ``"anthropic:claude-sonnet-4-20250514"``
            - ``"google-gla:gemini-2.0-flash"``

        Returns:
            Formatted model string.
        """
        config = PROVIDER_CONFIG.get(self.provider)
        prefix = config["model_prefix"] if config else self.provider
        return f"{prefix}:{self.model}"

    def get_config_summary(self) -> Dict[str, object]:
        """
        Return a serialisable summary of the agent's configuration.
        The API key is masked for safe display in UIs / logs.
        """
        masked = (
            f"{self.api_key[:6]}...{self.api_key[-4:]}"
            if len(self.api_key) > 12
            else "***"
        )
        return {
            "agent_name": self.agent_name,
            "agent_description": self.agent_description,
            "provider": self.provider,
            "model": self.model,
            "model_string": self.get_model_string(),
            "configured": self.is_configured(),
            "api_key_preview": masked,
        }

    # ══════════════════════════════════════════════════════════
    #  Protected helpers (used by subclasses)
    # ══════════════════════════════════════════════════════════

    def _set_api_key_env(self) -> None:
        """
        Inject the API key into the process environment.

        ``pydantic-ai`` reads provider keys from well-known env vars
        (``GROQ_API_KEY``, ``OPENAI_API_KEY``, …).  This method sets
        the correct variable right before an agent call so that keys
        supplied at runtime (e.g. from a Streamlit text-input) are
        picked up without requiring a restart.
        """
        if not self.api_key:
            logger.warning(
                "[%s] _set_api_key_env called but api_key is empty.",
                self.agent_name,
            )
            return

        config = PROVIDER_CONFIG.get(self.provider)
        if config:
            env_var = config["env_var"]
        else:
            # Best-effort fallback for unknown providers
            env_var = f"{self.provider.upper()}_API_KEY"
            logger.debug(
                "[%s] Unknown provider '%s'; using fallback env var '%s'.",
                self.agent_name,
                self.provider,
                env_var,
            )

        os.environ[env_var] = self.api_key
        logger.debug("[%s] Set %s for provider '%s'.", self.agent_name, env_var, self.provider)

    # ══════════════════════════════════════════════════════════
    #  Dunder helpers
    # ══════════════════════════════════════════════════════════

    def __repr__(self) -> str:
        status = "✓" if self.is_configured() else "✗"
        return (
            f"<{self.__class__.__name__} [{self.agent_name}] "
            f"{self.provider}:{self.model} ({status})>"
        )