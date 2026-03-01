"""
Validation helpers for action parameters and schemas.
"""

import pandas as pd
from typing import Dict, List, Optional, Any


def validate_columns_exist(df: pd.DataFrame, columns: List[str]) -> List[str]:
    """
    Check that all specified columns exist in the dataframe.
    Returns list of missing column names (empty if all valid).
    """
    existing = set(df.columns)
    missing = [col for col in columns if col not in existing]
    return missing


def validate_column_types(
    df: pd.DataFrame,
    columns: List[str],
    expected_type: str
) -> List[str]:
    """
    Check that columns match expected type category.
    expected_type: 'numeric', 'categorical', 'datetime', 'object'
    Returns list of columns that don't match.
    """
    type_map = {
        "numeric": lambda s: pd.api.types.is_numeric_dtype(s),
        "categorical": lambda s: s.dtype in ("object", "category"),
        "datetime": lambda s: pd.api.types.is_datetime64_any_dtype(s),
        "object": lambda s: s.dtype == "object",
        "bool": lambda s: pd.api.types.is_bool_dtype(s),
    }
    checker = type_map.get(expected_type)
    if not checker:
        return []

    invalid = [
        col for col in columns if col in df.columns and not checker(df[col])]
    return invalid


def validate_action_params(params: Dict, schema: Dict) -> List[str]:
    """
    Validate action parameters against a simple schema.
    Schema format:
    {
        "columns": {"type": "list", "required": True},
        "strategy": {"type": "string", "required": True, "choices": ["mean", "median"]},
        "fill_value": {"type": "any", "required": False}
    }
    Returns list of error messages.
    """
    errors = []
    for field_name, rules in schema.items():
        value = params.get(field_name)

        # Required check
        if rules.get("required", False) and value is None:
            errors.append(f"Missing required parameter: '{field_name}'")
            continue

        if value is None:
            continue

        # Type check
        expected = rules.get("type", "any")
        if expected == "list" and not isinstance(value, list):
            errors.append(f"Parameter '{field_name}' must be a list")
        elif expected == "string" and not isinstance(value, str):
            errors.append(f"Parameter '{field_name}' must be a string")
        elif expected == "number" and not isinstance(value, (int, float)):
            errors.append(f"Parameter '{field_name}' must be a number")
        elif expected == "bool" and not isinstance(value, bool):
            errors.append(f"Parameter '{field_name}' must be a boolean")

        # Choices check
        choices = rules.get("choices")
        if choices and value not in choices:
            errors.append(
                f"Parameter '{field_name}' must be one of {choices}, got '{value}'"
            )

        # Range check
        min_val = rules.get("min")
        max_val = rules.get("max")
        if min_val is not None and isinstance(value, (int, float)) and value < min_val:
            errors.append(f"Parameter '{field_name}' must be >= {min_val}")
        if max_val is not None and isinstance(value, (int, float)) and value > max_val:
            errors.append(f"Parameter '{field_name}' must be <= {max_val}")

    return errors
