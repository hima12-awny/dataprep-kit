"""
Datetime operations: extract components, calculate differences, parse dates.
"""

import pandas as pd
from typing import Dict, List, Any

from actions.base import BaseAction
from config.registry import register_action
from utils.validators import validate_columns_exist


@register_action("datetime_ops")
class DatetimeOpsAction(BaseAction):
    action_type = "datetime_ops"
    domain = "conversion"
    display_name = "Datetime Operations"

    OPERATIONS = [
        "extract_components",
        "date_diff",
        "to_unix_timestamp",
        "from_unix_timestamp",
    ]

    COMPONENTS = [
        "year", "month", "day", "hour", "minute", "second",
        "weekday", "day_name", "month_name", "quarter",
        "week", "day_of_year", "is_weekend", "is_month_start",
        "is_month_end",
    ]

    def validate(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> List[str]:
        errors = []

        operation = parameters.get("operation")
        if not operation:
            errors.append("Missing required parameter: 'operation'")
            return errors

        if operation not in self.OPERATIONS:
            errors.append(f"Invalid operation '{operation}'")

        column = parameters.get("column")
        if not column:
            errors.append("Missing required parameter: 'column'")
            return errors

        if column not in df.columns:
            errors.append(f"Column '{column}' not found")

        if operation == "extract_components":
            components = parameters.get("components", [])
            if not components:
                errors.append(
                    "'extract_components' requires 'components' list")
            invalid = [c for c in components if c not in self.COMPONENTS]
            if invalid:
                errors.append(f"Invalid components: {invalid}")

        if operation == "date_diff":
            column2 = parameters.get("column2")
            if not column2:
                errors.append("'date_diff' requires 'column2' parameter")
            elif column2 not in df.columns:
                errors.append(f"Column '{column2}' not found")

        return errors

    def execute(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
        result = df.copy()
        operation = parameters["operation"]
        column = parameters["column"]

        if operation == "extract_components":
            components = parameters["components"]
            # Ensure column is datetime
            if not pd.api.types.is_datetime64_any_dtype(result[column]):
                result[column] = pd.to_datetime(
                    result[column], errors="coerce")

            prefix = parameters.get("prefix", column)
            dt = result[column].dt

            component_map = {
                "year": dt.year,
                "month": dt.month,
                "day": dt.day,
                "hour": dt.hour,
                "minute": dt.minute,
                "second": dt.second,
                "weekday": dt.weekday,
                "day_name": dt.day_name(),
                "month_name": dt.month_name(),
                "quarter": dt.quarter,
                "week": dt.isocalendar().week.astype(int),
                "day_of_year": dt.day_of_year,
                "is_weekend": (dt.weekday >= 5).astype(int),
                "is_month_start": dt.is_month_start.astype(int),
                "is_month_end": dt.is_month_end.astype(int),
            }

            for comp in components:
                if comp in component_map:
                    result[f"{prefix}_{comp}"] = component_map[comp]

        elif operation == "date_diff":
            column2 = parameters["column2"]
            unit = parameters.get("unit", "days")
            new_col = parameters.get("new_column", f"{column}_minus_{column2}")

            col1 = pd.to_datetime(result[column], errors="coerce")
            col2 = pd.to_datetime(result[column2], errors="coerce")
            diff = col1 - col2

            if unit == "days":
                result[new_col] = diff.dt.days
            elif unit == "hours":
                result[new_col] = diff.dt.total_seconds() / 3600
            elif unit == "seconds":
                result[new_col] = diff.dt.total_seconds()

        elif operation == "to_unix_timestamp":
            new_col = parameters.get("new_column", f"{column}_unix")
            dt_col = pd.to_datetime(result[column], errors="coerce")
            result[new_col] = (
                dt_col - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")

        elif operation == "from_unix_timestamp":
            new_col = parameters.get("new_column", f"{column}_datetime")
            result[new_col] = pd.to_datetime(
                result[column], unit="s", errors="coerce")

        return result

    def get_parameter_schema(self) -> Dict:
        return {
            "operation": {
                "type": "string",
                "required": True,
                "choices": self.OPERATIONS,
                "description": "Datetime operation to perform.",
            },
            "column": {
                "type": "string",
                "required": True,
                "description": "Source datetime column.",
            },
            "components": {
                "type": "list",
                "required": False,
                "choices": self.COMPONENTS,
                "description": "Components to extract.",
            },
            "column2": {
                "type": "string",
                "required": False,
                "description": "Second column for date_diff.",
            },
            "unit": {
                "type": "string",
                "required": False,
                "default": "days",
                "choices": ["days", "hours", "seconds"],
                "description": "Unit for date_diff result.",
            },
            "prefix": {
                "type": "string",
                "required": False,
                "description": "Prefix for new component columns.",
            },
            "new_column": {
                "type": "string",
                "required": False,
                "description": "Name for the new column.",
            },
        }

    def get_description(self, parameters: Dict[str, Any]) -> str:
        operation = parameters.get("operation", "unknown")
        column = parameters.get("column", "?")
        if operation == "extract_components":
            comps = parameters.get("components", [])
            return f"Extract [{', '.join(comps[:4])}] from '{column}'"
        elif operation == "date_diff":
            return f"Calculate date difference: '{column}' - '{parameters.get('column2')}'"
        return f"Datetime operation '{operation}' on '{column}'"

    def get_code_snippet(self, parameters: Dict[str, Any]) -> str:
        operation = parameters.get("operation")
        column = parameters.get("column")
        if operation == "extract_components":
            return f"df['{column}_year'] = df['{column}'].dt.year  # etc."
        elif operation == "date_diff":
            return f"df['diff'] = (df['{column}'] - df['{parameters.get('column2')}']).dt.days"
        return f"# Datetime: {operation}"
