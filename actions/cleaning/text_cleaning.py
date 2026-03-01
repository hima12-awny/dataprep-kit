"""
Text/string cleaning operations: trim, case, regex, special characters.
"""

import re
import pandas as pd
from typing import Dict, List, Any

from actions.base import BaseAction
from config.registry import register_action
from utils.validators import validate_columns_exist


@register_action("text_cleaning")
class TextCleaningAction(BaseAction):
    action_type = "text_cleaning"
    domain = "cleaning"
    display_name = "Text Cleaning"

    OPERATIONS = [
        "trim_whitespace",
        "lowercase",
        "uppercase",
        "titlecase",
        "remove_special_chars",
        "remove_punctuation",
        "regex_replace",
        "strip_html",
        "collapse_whitespace",
    ]

    def validate(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> List[str]:
        errors = []

        columns = parameters.get("columns", [])
        if not columns:
            errors.append("At least one column is required")
            return errors

        missing = validate_columns_exist(df, columns)
        if missing:
            errors.append(f"Columns not found: {missing}")

        operations = parameters.get("operations", [])
        if not operations:
            errors.append("At least one operation is required")

        for op in operations:
            if op not in self.OPERATIONS:
                errors.append(
                    f"Invalid operation '{op}'. Must be one of {self.OPERATIONS}")

        if "regex_replace" in operations:
            if not parameters.get("regex_pattern"):
                errors.append(
                    "'regex_replace' requires 'regex_pattern' parameter")

        return errors

    def execute(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
        result = df.copy()
        columns = parameters["columns"]
        operations = parameters["operations"]

        for col in columns:
            if col not in result.columns or result[col].dtype != "object":
                continue

            series = result[col].copy()

            for op in operations:
                non_null_mask = series.notna()

                if op == "trim_whitespace":
                    series[non_null_mask] = series[
                        non_null_mask
                    ].str.strip()  # type: ignore

                elif op == "lowercase":
                    series[non_null_mask] = series[
                        non_null_mask
                    ].str.lower()  # type: ignore

                elif op == "uppercase":
                    series[non_null_mask] = series[
                        non_null_mask
                    ].str.upper()  # type: ignore

                elif op == "titlecase":
                    series[non_null_mask] = series[
                        non_null_mask
                    ].str.title()  # type: ignore

                elif op == "remove_special_chars":
                    series[non_null_mask] = series[
                        non_null_mask
                    ].str.replace(  # type: ignore
                        r"[^a-zA-Z0-9\s]", "", regex=True
                    )

                elif op == "remove_punctuation":
                    series[non_null_mask] = series[
                        non_null_mask
                    ].str.replace(  # type: ignore
                        r"[^\w\s]", "", regex=True
                    )

                elif op == "regex_replace":
                    pattern = parameters.get("regex_pattern", "")
                    replacement = parameters.get("regex_replacement", "")
                    series[non_null_mask] = series[non_null_mask].str.replace(  # type: ignore
                        pattern, replacement, regex=True
                    )

                elif op == "strip_html":
                    series[non_null_mask] = series[non_null_mask].str.replace(  # type: ignore
                        r"<[^>]+>", "", regex=True
                    )

                elif op == "collapse_whitespace":
                    series[non_null_mask] = series[non_null_mask].str.replace(  # type: ignore
                        r"\s+", " ", regex=True
                    ).str.strip()

            result[col] = series

        return result

    def get_parameter_schema(self) -> Dict:
        return {
            "columns": {
                "type": "list",
                "required": True,
                "description": "Text columns to clean.",
            },
            "operations": {
                "type": "list",
                "required": True,
                "choices": self.OPERATIONS,
                "description": "Cleaning operations to apply (in order).",
            },
            "regex_pattern": {
                "type": "string",
                "required": False,
                "default": None,
                "description": "Regex pattern for 'regex_replace' operation.",
            },
            "regex_replacement": {
                "type": "string",
                "required": False,
                "default": "",
                "description": "Replacement string for 'regex_replace' operation.",
            },
        }

    def get_description(self, parameters: Dict[str, Any]) -> str:
        columns = parameters.get("columns", [])
        operations = parameters.get("operations", [])
        col_str = ", ".join(columns[:3])
        op_str = ", ".join(operations[:3])
        return f"Text cleaning [{op_str}] on columns [{col_str}]"

    def get_code_snippet(self, parameters: Dict[str, Any]) -> str:
        lines = []
        columns = parameters.get("columns", [])
        operations = parameters.get("operations", [])
        for op in operations:
            if op == "trim_whitespace":
                lines.append(
                    f"df[{columns}] = df[{columns}].apply(lambda x: x.str.strip())")
            elif op == "lowercase":
                lines.append(
                    f"df[{columns}] = df[{columns}].apply(lambda x: x.str.lower())")
        return "\n".join(lines) if lines else f"# Text cleaning: {operations}"
