"""
Detect and remove duplicate rows.
"""

import pandas as pd
from typing import Dict, List, Any

from actions.base import BaseAction
from config.registry import register_action
from utils.validators import validate_columns_exist


@register_action("handle_duplicates")
class HandleDuplicatesAction(BaseAction):
    action_type = "handle_duplicates"
    domain = "cleaning"
    display_name = "Handle Duplicates"
    description_template = "Remove duplicate rows keeping '{keep}'"

    KEEP_OPTIONS = ["first", "last", "none"]

    def validate(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> List[str]:
        errors = []

        keep = parameters.get("keep", "first")
        if keep not in self.KEEP_OPTIONS and keep is not False:
            errors.append(
                f"Invalid 'keep' value: '{keep}'. Must be one of {self.KEEP_OPTIONS}")

        subset = parameters.get("subset")
        if subset:
            missing = validate_columns_exist(df, subset)
            if missing:
                errors.append(f"Columns not found: {missing}")

        return errors

    def execute(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
        result = df.copy()
        subset = parameters.get("subset")
        keep = parameters.get("keep", "first")

        # Map "none" string to False for pandas
        if keep == "none":
            keep = False

        result = result.drop_duplicates(
            subset=subset, keep=keep).reset_index(drop=True)
        return result

    def preview(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> Dict:
        subset = parameters.get("subset")
        keep = parameters.get("keep", "first")
        if keep == "none":
            keep = False

        duplicates = df[df.duplicated(subset=subset, keep=False)]
        result_df = df.drop_duplicates(
            subset=subset, keep=keep).reset_index(drop=True)

        return {
            "before": df.head(50),
            "after": result_df.head(50),
            "summary": f"Found {len(duplicates)} duplicate rows. Will remove {len(df) - len(result_df)} rows.",
            "affected_columns": subset or df.columns.tolist(),
            "rows_before": len(df),
            "rows_after": len(result_df),
            "cols_before": len(df.columns),
            "cols_after": len(result_df.columns),
            "duplicate_preview": duplicates.head(20),
            "duplicate_count": len(duplicates),
        }

    def get_parameter_schema(self) -> Dict:
        return {
            "subset": {
                "type": "list",
                "required": False,
                "default": None,
                "description": "Columns to consider for duplicates. None = all columns.",
            },
            "keep": {
                "type": "string",
                "required": False,
                "default": "first",
                "choices": self.KEEP_OPTIONS,
                "description": "Which duplicate to keep: 'first', 'last', or 'none' (remove all).",
            },
        }

    def get_description(self, parameters: Dict[str, Any]) -> str:
        keep = parameters.get("keep", "first")
        subset = parameters.get("subset")
        scope = f"based on {subset}" if subset else "across all columns"
        return f"Remove duplicate rows {scope}, keeping '{keep}'"

    def get_code_snippet(self, parameters: Dict[str, Any]) -> str:
        subset = parameters.get("subset")
        keep = parameters.get("keep", "first")
        if keep == "none":
            keep = False
        subset_str = f"subset={subset}, " if subset else ""
        return f"df = df.drop_duplicates({subset_str}keep={repr(keep)}).reset_index(drop=True)"
