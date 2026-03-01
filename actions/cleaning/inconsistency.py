"""
Handle data inconsistencies: value mapping, merging rare categories, standardization.
"""

import pandas as pd
from typing import Dict, List, Any

from actions.base import BaseAction
from config.registry import register_action
from utils.validators import validate_columns_exist


@register_action("inconsistency")
class InconsistencyAction(BaseAction):
    action_type = "inconsistency"
    domain = "cleaning"
    display_name = "Fix Inconsistencies"

    OPERATIONS = [
        "value_mapping",
        "merge_rare_categories",
        "standardize_values",
    ]

    def validate(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> List[str]:
        errors = []

        operation = parameters.get("operation")
        if not operation:
            errors.append("Missing required parameter: 'operation'")
            return errors

        if operation not in self.OPERATIONS:
            errors.append(
                f"Invalid operation '{operation}'. Must be one of {self.OPERATIONS}")

        columns = parameters.get("columns", [])
        if not columns:
            errors.append("At least one column is required")
            return errors

        missing = validate_columns_exist(df, columns)
        if missing:
            errors.append(f"Columns not found: {missing}")

        if operation == "value_mapping":
            mapping = parameters.get("mapping")
            if not mapping or not isinstance(mapping, dict):
                errors.append(
                    "'value_mapping' requires a 'mapping' dict parameter")

        return errors

    def execute(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
        result = df.copy()
        operation = parameters["operation"]
        columns = parameters["columns"]

        if operation == "value_mapping":
            mapping = parameters["mapping"]
            for col in columns:
                if col in result.columns:
                    result[col] = result[col].replace(mapping)

        elif operation == "merge_rare_categories":
            threshold = parameters.get("threshold", 0.01)
            replacement = parameters.get("replacement", "Other")
            for col in columns:
                if col in result.columns:
                    freq = result[col].value_counts(normalize=True)
                    rare_values = freq[
                        freq < threshold
                    ].index.tolist()  # type: ignore
                    result[col] = result[col].replace(rare_values, replacement)

        elif operation == "standardize_values":
            # Trim + lowercase for consistent comparison
            case = parameters.get("case", "lower")
            for col in columns:
                if col in result.columns and result[col].dtype == "object":
                    result[col] = result[col].str.strip()
                    if case == "lower":
                        result[col] = result[col].str.lower()
                    elif case == "upper":
                        result[col] = result[col].str.upper()
                    elif case == "title":
                        result[col] = result[col].str.title()

        return result

    def get_parameter_schema(self) -> Dict:
        return {
            "columns": {
                "type": "list",
                "required": True,
                "description": "Columns to fix.",
            },
            "operation": {
                "type": "string",
                "required": True,
                "choices": self.OPERATIONS,
                "description": "Type of inconsistency fix.",
            },
            "mapping": {
                "type": "any",
                "required": False,
                "default": None,
                "description": "Dict mapping old values to new values (for 'value_mapping').",
            },
            "threshold": {
                "type": "number",
                "required": False,
                "default": 0.01,
                "min": 0,
                "max": 1,
                "description": "Frequency threshold for 'merge_rare_categories'.",
            },
            "replacement": {
                "type": "string",
                "required": False,
                "default": "Other",
                "description": "Replacement value for rare categories.",
            },
            "case": {
                "type": "string",
                "required": False,
                "default": "lower",
                "choices": ["lower", "upper", "title"],
                "description": "Case for 'standardize_values'.",
            },
        }

    def get_description(self, parameters: Dict[str, Any]) -> str:
        operation = parameters.get("operation", "unknown")
        columns = parameters.get("columns", [])
        col_str = ", ".join(columns[:3])
        return f"Fix inconsistencies ({operation}) in [{col_str}]"

    def get_code_snippet(self, parameters: Dict[str, Any]) -> str:
        operation = parameters.get("operation")
        if operation == "value_mapping":
            return f"df[columns] = df[columns].replace({parameters.get('mapping', {})})"
        elif operation == "merge_rare_categories":
            return "# Merge categories with frequency below threshold into 'Other'"
        return f"# Fix inconsistencies: {operation}"
