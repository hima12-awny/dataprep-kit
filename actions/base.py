"""
BaseAction — abstract base class for all data transformation actions.
Every action in the system (cleaning, conversion, feature engineering) inherits from this.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd

from utils.id_generator import generate_action_id


class BaseAction(ABC):
    """
    Abstract base class for all pipeline actions.

    Subclasses must implement:
        - action_type (class attribute)
        - domain (class attribute)
        - display_name (class attribute)
        - validate()
        - execute()
        - get_parameter_schema()

    Subclasses may override:
        - preview()    → default implementation runs execute on a copy
        - get_description()
        - get_code_snippet()
    """

    # ── Class Attributes (set by subclasses or @register_action) ──
    action_type: str = "base"
    domain: str = "base"  # "cleaning", "conversion", "feature_engineering"
    display_name: str = "Base Action"
    description_template: str = ""

    # ── Validation ────────────────────────────────────────────

    @abstractmethod
    def validate(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> List[str]:
        """
        Validate parameters against the current dataframe state.

        Args:
            df: The current dataframe.
            parameters: The action parameters dict.

        Returns:
            List of error message strings. Empty list = valid.
        """
        pass

    # ── Preview ───────────────────────────────────────────────

    def preview(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> Dict:
        """
        Preview the action result without mutating the original dataframe.

        Returns:
            {
                "before": pd.DataFrame (sample),
                "after": pd.DataFrame (sample),
                "summary": str,
                "affected_columns": list,
                "rows_before": int,
                "rows_after": int,
                "cols_before": int,
                "cols_after": int,
            }
        """
        df_copy = df.copy()
        result_df = self.execute(df_copy, parameters)

        # Determine which columns were affected
        affected_columns = self._detect_affected_columns(df, result_df)

        # Build preview with limited rows
        preview_rows = min(50, len(df))
        return {
            "before": df.head(preview_rows),
            "after": result_df.head(preview_rows),
            "summary": self._generate_summary(df, result_df, parameters),
            "affected_columns": affected_columns,
            "rows_before": len(df),
            "rows_after": len(result_df),
            "cols_before": len(df.columns),
            "cols_after": len(result_df.columns),
        }

    # ── Execution ─────────────────────────────────────────────

    @abstractmethod
    def execute(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
        """
        Execute the transformation on the dataframe.

        IMPORTANT: Must NOT mutate the input df. Return a new dataframe.

        Args:
            df: The current dataframe.
            parameters: The action parameters dict.

        Returns:
            A new, transformed pd.DataFrame.
        """
        pass

    # ── Serialization ─────────────────────────────────────────

    def serialize(
        self,
        parameters: Dict[str, Any],
        author: str = "user",
        description: Optional[str] = None,
        action_id: Optional[str] = None,
    ) -> Dict:
        """
        Serialize this action invocation to the standard JSON action schema.
        """
        return {
            "action_id": action_id or generate_action_id(),
            "action_type": self.action_type,
            "description": description or self.get_description(parameters),
            "author": author,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "parameters": parameters,
            "preview_only": False,
        }

    @classmethod
    def deserialize(cls, data: Dict) -> Tuple["BaseAction", Dict[str, Any]]:
        """
        Reconstruct an action instance from a serialized dict.
        Returns (action_instance, parameters).
        """
        from config.registry import ActionRegistry

        action_class = ActionRegistry.get(data["action_type"])
        if not action_class:
            raise ValueError(f"Unknown action_type: {data['action_type']}")
        return action_class(), data.get("parameters", {})

    # ── Parameter Schema ──────────────────────────────────────

    @abstractmethod
    def get_parameter_schema(self) -> Dict:
        """
        Return the parameter schema for this action.
        Used by the UI to render forms and by validation logic.

        Format:
        {
            "param_name": {
                "type": "string" | "number" | "list" | "bool" | "any",
                "required": True/False,
                "default": value,
                "choices": [...],  # optional
                "description": "...",
                "min": number,  # optional
                "max": number,  # optional
            }
        }
        """
        pass

    # ── Description & Code ────────────────────────────────────

    def get_description(self, parameters: Dict[str, Any]) -> str:
        """
        Generate a human-readable description for this action with given parameters.
        Override in subclasses for custom descriptions.
        """
        if self.description_template:
            try:
                return self.description_template.format(**parameters)
            except (KeyError, IndexError):
                pass
        return f"{self.display_name} on {parameters.get('columns', 'dataset')}"

    def get_code_snippet(self, parameters: Dict[str, Any]) -> str:
        """
        Generate reproducible Python/pandas code for this action.
        Override in subclasses.
        """
        return f"# {self.display_name}\n# Parameters: {parameters}"

    # ── Private Helpers ───────────────────────────────────────

    @staticmethod
    def _detect_affected_columns(
        before: pd.DataFrame, after: pd.DataFrame
    ) -> List[str]:
        """Detect which columns were changed between before and after."""
        affected = []
        common_cols = set(before.columns) & set(after.columns)

        for col in common_cols:
            try:
                if not before[col].equals(after[col]):
                    affected.append(col)
            except Exception:
                affected.append(col)

        # New columns
        new_cols = set(after.columns) - set(before.columns)
        affected.extend(list(new_cols))

        # Removed columns
        removed_cols = set(before.columns) - set(after.columns)
        affected.extend(list(removed_cols))

        return affected

    def _generate_summary(
        self,
        before: pd.DataFrame,
        after: pd.DataFrame,
        parameters: Dict[str, Any],
    ) -> str:
        """Generate a summary of what changed."""
        parts = []

        row_diff = len(after) - len(before)
        if row_diff != 0:
            direction = "added" if row_diff > 0 else "removed"
            parts.append(f"{abs(row_diff)} rows {direction}")

        col_diff = len(after.columns) - len(before.columns)
        if col_diff != 0:
            direction = "added" if col_diff > 0 else "removed"
            parts.append(f"{abs(col_diff)} columns {direction}")

        null_before = before.isna().sum().sum()
        null_after = after.isna().sum().sum()
        null_diff = null_after - null_before
        if null_diff != 0:
            direction = "introduced" if null_diff > 0 else "resolved"
            parts.append(f"{abs(null_diff)} null values {direction}")

        if not parts:
            parts.append("Data values modified in place")

        return "; ".join(parts)