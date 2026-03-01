"""
Type casting: convert columns to int, float, string, bool, datetime, category.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any

from actions.base import BaseAction
from config.registry import register_action
from utils.validators import validate_columns_exist


@register_action("type_casting")
class TypeCastingAction(BaseAction):
    action_type = "type_casting"
    domain = "conversion"
    display_name = "Type Casting"

    TARGET_TYPES = [
        "int8", "int16", "int32", "int64", "Int8", "Int16", "Int32", "Int64",
        "float16", "float32", "float64", "Float32", "Float64",
        "string", "object", "bool", "boolean",
        "datetime64[ns]", "datetime64[ms]", "datetime64[s]",
        "category"
    ]

    def validate(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> List[str]:
        errors = []

        conversions = parameters.get("conversions", {})
        if not conversions:
            errors.append("At least one column-to-type conversion is required")
            return errors

        columns = list(conversions.keys())
        missing = validate_columns_exist(df, columns)
        if missing:
            errors.append(f"Columns not found: {missing}")

        for col, target_type in conversions.items():
            if target_type not in self.TARGET_TYPES:
                errors.append(
                    f"Invalid target type '{target_type}' for column '{col}'")

        return errors

    def execute(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
        result = df.copy()
        conversions = parameters["conversions"]
        errors_handling = parameters.get(
            "errors", "coerce")  # "coerce", "raise", "ignore"
        date_format = parameters.get("date_format")  # For datetime conversion

        for col, target_type in conversions.items():
            if col not in result.columns:
                continue

            try:
                # Integer types
                if target_type in ["int8", "int16", "int32", "int64"]:
                    result[col] = pd.to_numeric(
                        result[col], errors=errors_handling)
                    if errors_handling == "coerce":
                        # Use pandas nullable integer type if available
                        nullable_type = (
                            "Int8" if target_type == "int8"
                            else "Int16" if target_type == "int16"
                            else "Int32" if target_type == "int32"
                            else "Int64"  # Default
                        )
                        result[col] = result[col].astype(nullable_type)
                    else:
                        result[col] = result[col].astype(target_type)

                # Pandas nullable integer types
                elif target_type in ["Int8", "Int16", "Int32", "Int64"]:
                    result[col] = pd.to_numeric(
                        result[col], errors=errors_handling
                    ).astype(target_type)  # type: ignore

                # Float types
                elif target_type in ["float16", "float32", "float64", "Float32", "Float64"]:
                    # Map string type to canonical numpy type for .astype()
                    astype_type = (
                        "float16" if target_type in ["float16"]
                        else "float32" if target_type in ["float32", "Float32"]
                        # float64/Float64 (np.float64 allows both)
                        else "float64"
                    )
                    result[col] = pd.to_numeric(
                        result[col],
                        errors=errors_handling).astype(astype_type)  # type: ignore

                # String/Object types
                elif target_type in ["string"]:
                    # Use pandas StringDtype for "string", not Python str
                    result[col] = result[col].astype("string")
                elif target_type == "object":
                    result[col] = result[col].astype("object")

                # Boolean types
                elif target_type in ["bool", "boolean"]:
                    # Pandas BooleanDtype if "boolean"
                    bool_map = {
                        "true": True, "false": False,
                        "yes": True, "no": False,
                        "1": True, "0": False,
                        "t": True, "f": False,
                        "y": True, "n": False,
                    }
                    if result[col].dtype == "object":
                        converted = result[col].astype(
                            str).str.strip().str.lower().map(bool_map)
                        if target_type == "boolean":
                            result[col] = converted.astype("boolean")
                        else:
                            result[col] = converted.astype(bool)
                    else:
                        if target_type == "boolean":
                            result[col] = result[col].astype("boolean")
                        else:
                            result[col] = result[col].astype(bool)

                # Datetime types (different resolutions)
                elif target_type in ["datetime64[ns]", "datetime64[ms]", "datetime64[s]"]:
                    if date_format:
                        result[col] = pd.to_datetime(
                            result[col], format=date_format, errors=errors_handling)
                    else:
                        result[col] = pd.to_datetime(
                            result[col], errors=errors_handling)
                    # Downcast to lower precision if needed
                    if target_type != "datetime64[ns]":
                        result[col] = result[col].astype(target_type)

                # Category
                elif target_type == "category":
                    result[col] = result[col].astype("category")

            except Exception as e:
                if errors_handling == "raise":
                    raise
                # On coerce/ignore, skip silently

        return result

    def get_parameter_schema(self) -> Dict:
        return {
            "conversions": {
                "type": "any",
                "required": True,
                "description": f"Dict mapping column names to target types from this {self.TARGET_TYPES}. E.g. {{'age': 'int64', 'date': 'datetime64[ns]'}}",
            },
            "errors": {
                "type": "string",
                "required": False,
                "default": "coerce",
                "choices": ["coerce", "raise", "ignore"],
                "description": "How to handle conversion errors.",
            },
            "date_format": {
                "type": "string",
                "required": False,
                "default": None,
                "description": "Explicit datetime format string (e.g. '%Y-%m-%d').",
            },
        }

    def get_description(self, parameters: Dict[str, Any]) -> str:
        conversions = parameters.get("conversions", {})
        parts = [
            f"'{col}' → {dtype}" for col,
            dtype in list(conversions.items())[:3]
        ]
        desc = ", ".join(parts)
        if len(conversions) > 3:
            desc += f" (+{len(conversions)-3} more)"
        return f"Type casting: {desc}"

    def get_code_snippet(self, parameters: Dict[str, Any]) -> str:
        conversions = parameters.get("conversions", {})
        lines = []
        for col, target in conversions.items():
            if target == "datetime64[ns]":
                lines.append(
                    f"df['{col}'] = pd.to_datetime(df['{col}'], errors='coerce')")
            elif target in ("int64", "float64"):
                lines.append(
                    f"df['{col}'] = pd.to_numeric(df['{col}'], errors='coerce')")
            else:
                lines.append(f"df['{col}'] = df['{col}'].astype('{target}')")
        return "\n".join(lines)
