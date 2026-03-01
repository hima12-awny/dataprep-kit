"""
Central registry that maps action_type strings to their Action classes.
This enables dynamic deserialization of pipelines from JSON.
"""

from typing import Dict, Type, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from actions.base import BaseAction


class _ActionRegistry:
    """Singleton registry for action types."""

    def __init__(self):
        self._registry: Dict[str, Type["BaseAction"]] = {}
        self._initialized: bool = False

    def register(self, action_type: str, action_class: Type["BaseAction"]):
        """Register an action class for a given action_type string."""
        self._registry[action_type] = action_class

    def get(self, action_type: str) -> Optional[Type["BaseAction"]]:
        """Retrieve an action class by its type string."""
        self._ensure_initialized()
        return self._registry.get(action_type)

    def get_all(self) -> Dict[str, Type["BaseAction"]]:
        """Return the full registry."""
        self._ensure_initialized()
        return dict(self._registry)

    def get_by_domain(self, domain: str) -> Dict[str, Type["BaseAction"]]:
        """Return actions filtered by domain (cleaning, conversion, feature_engineering)."""
        self._ensure_initialized()
        return {
            k: v for k, v in self._registry.items()
            if getattr(v, "domain", None) == domain
        }

    def _ensure_initialized(self):
        """Lazy-load all action modules to populate the registry."""
        if not self._initialized:
            self._initialized = True
            self._load_all_actions()

    def _load_all_actions(self):
        """Import all action modules to trigger their @register decorators."""
        # Cleaning
        try:
            import actions.cleaning.handle_missing
            import actions.cleaning.handle_duplicates
            import actions.cleaning.handle_outliers
            import actions.cleaning.text_cleaning
            import actions.cleaning.inconsistency
        except ImportError:
            pass

        # Conversion
        try:
            import actions.conversion.type_casting
            import actions.conversion.datetime_ops
            import actions.conversion.numeric_transforms
            import actions.conversion.encoding
        except ImportError:
            pass

        # Feature Engineering
        try:
            import actions.feature_engineering.column_ops
            import actions.feature_engineering.aggregation
            import actions.feature_engineering.temporal
            import actions.feature_engineering.interaction
        except ImportError:
            pass


# Singleton
ActionRegistry = _ActionRegistry()


def register_action(action_type: str):
    """
    Decorator to register an action class in the global registry.

    Usage:
        @register_action("handle_missing")
        class HandleMissingAction(BaseAction):
            ...
    """
    def decorator(cls):
        cls.action_type = action_type
        ActionRegistry.register(action_type, cls)
        return cls
    return decorator