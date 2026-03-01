"""
Streamlit session state manager.
Single source of truth for the current dataset, pipeline, and UI state.
"""

import streamlit as st
import pandas as pd
from typing import Optional, Dict, Any

from core.dataset import Dataset
from core.pipeline import Pipeline


class StateManager:
    """
    Manages all session state for the Streamlit app.
    Wraps st.session_state with typed accessors.
    """

    # ── Keys ──────────────────────────────────────────────────
    DATASET_KEY = "dataset"
    PIPELINE_KEY = "pipeline"
    RECOMMENDATIONS_KEY = "recommendations"
    CURRENT_PAGE_KEY = "current_page"
    NOTIFICATIONS_KEY = "notifications"

    @classmethod
    def initialize(cls):
        """Initialize all session state keys with defaults. Safe to call multiple times."""
        defaults = {
            cls.DATASET_KEY: None,
            cls.PIPELINE_KEY: Pipeline(),
            cls.RECOMMENDATIONS_KEY: [],
            cls.CURRENT_PAGE_KEY: "Import",
            cls.NOTIFICATIONS_KEY: [],
            "profiling_cache": None,
            "last_action_result": None,
        }
        for key, default in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default

    # ── Dataset ───────────────────────────────────────────────

    @classmethod
    def get_dataset(cls) -> Optional[Dataset]:
        """Get the current Dataset object."""
        return st.session_state.get(cls.DATASET_KEY)

    @classmethod
    def set_dataset(cls, dataset: Dataset):
        """Set the dataset and reset dependent state."""
        st.session_state[cls.DATASET_KEY] = dataset
        st.session_state["profiling_cache"] = None
        st.session_state[cls.RECOMMENDATIONS_KEY] = []

    @classmethod
    def has_dataset(cls) -> bool:
        """Check if a dataset has been loaded."""
        return st.session_state.get(cls.DATASET_KEY) is not None

    @classmethod
    def get_dataframe(cls) -> Optional[pd.DataFrame]:
        """Shortcut to get the current dataframe."""
        dataset = cls.get_dataset()
        return dataset.df if dataset else None

    # ── Pipeline ──────────────────────────────────────────────

    @classmethod
    def get_pipeline(cls) -> Pipeline:
        """Get the current pipeline."""
        return st.session_state.get(cls.PIPELINE_KEY, Pipeline())

    @classmethod
    def set_pipeline(cls, pipeline: Pipeline):
        """Set the pipeline."""
        st.session_state[cls.PIPELINE_KEY] = pipeline

    # ── Recommendations ───────────────────────────────────────

    @classmethod
    def get_recommendations(cls) -> list:
        """Get the current list of recommendation action dicts."""
        return st.session_state.get(cls.RECOMMENDATIONS_KEY, [])

    @classmethod
    def set_recommendations(cls, recommendations: list):
        """Set recommendations."""
        st.session_state[cls.RECOMMENDATIONS_KEY] = recommendations

    @classmethod
    def add_recommendation(cls, recommendation: Dict):
        """Add a single recommendation."""
        recs = cls.get_recommendations()
        recs.append(recommendation)
        st.session_state[cls.RECOMMENDATIONS_KEY] = recs

    @classmethod
    def remove_recommendation(cls, action_id: str):
        """Remove a recommendation by action_id."""
        recs = cls.get_recommendations()
        st.session_state[cls.RECOMMENDATIONS_KEY] = [
            r for r in recs if r.get("action_id") != action_id
        ]

    # ── Notifications ─────────────────────────────────────────

    @classmethod
    def add_notification(cls, message: str, level: str = "info"):
        """Add a notification to display."""
        notifs = st.session_state.get(cls.NOTIFICATIONS_KEY, [])
        notifs.append({"message": message, "level": level})
        st.session_state[cls.NOTIFICATIONS_KEY] = notifs

    @classmethod
    def get_notifications(cls) -> list:
        """Get and clear all pending notifications."""
        notifs = st.session_state.get(cls.NOTIFICATIONS_KEY, [])
        st.session_state[cls.NOTIFICATIONS_KEY] = []
        return notifs

    # ── Profiling Cache ───────────────────────────────────────

    @classmethod
    def get_profiling_cache(cls) -> Optional[Dict]:
        """Get cached profiling results."""
        return st.session_state.get("profiling_cache")

    @classmethod
    def set_profiling_cache(cls, cache: Dict):
        """Set profiling cache."""
        st.session_state["profiling_cache"] = cache

    @classmethod
    def invalidate_profiling_cache(cls):
        """Invalidate profiling cache (call after any data transformation)."""
        st.session_state["profiling_cache"] = None

    # ── Last Action Result ────────────────────────────────────

    @classmethod
    def set_last_action_result(cls, result: Dict):
        """Store the result of the last executed action."""
        st.session_state["last_action_result"] = result

    @classmethod
    def get_last_action_result(cls) -> Optional[Dict]:
        """Get the result of the last executed action."""
        result = st.session_state.get("last_action_result")
        st.session_state["last_action_result"] = None
        return result

    # ── AI Configuration ──────────────────────────────────────

    AI_CONFIG_KEY = "ai_config"

    @classmethod
    def get_ai_config(cls) -> dict:
        """Get AI configuration."""
        defaults = {
            "provider": "groq",
            "model": "llama-3.3-70b-versatile",
            "api_key": "",
            "data_description": None,  # DatasetDescription dict or None
            "data_description_text": "",  # raw user-editable text
            "target_tracks": [],
        }
        config = st.session_state.get(cls.AI_CONFIG_KEY, {})
        # Merge with defaults
        for key, val in defaults.items():
            config.setdefault(key, val)
        return config

    @classmethod
    def set_ai_config(cls, config: dict):
        """Set AI configuration."""
        st.session_state[cls.AI_CONFIG_KEY] = config

    @classmethod
    def update_ai_config(cls, **kwargs):
        """Update specific AI config fields."""
        config = cls.get_ai_config()
        config.update(kwargs)
        st.session_state[cls.AI_CONFIG_KEY] = config

    @classmethod
    def is_ai_configured(cls) -> bool:
        """Check if AI agent is ready to use."""
        config = cls.get_ai_config()
        return bool(config.get("api_key"))

    @classmethod
    def has_data_description(cls) -> bool:
        """Check if data description has been provided."""
        config = cls.get_ai_config()
        return bool(config.get("data_description"))

    # ── Reset ─────────────────────────────────────────────────

    @classmethod
    def reset_all(cls):
        """Reset all state to defaults."""
        keys_to_reset = [
            cls.DATASET_KEY,
            cls.PIPELINE_KEY,
            cls.RECOMMENDATIONS_KEY,
            cls.NOTIFICATIONS_KEY,
            "profiling_cache",
            "last_action_result",
        ]
        for key in keys_to_reset:
            if key in st.session_state:
                del st.session_state[key]
        cls.initialize()
