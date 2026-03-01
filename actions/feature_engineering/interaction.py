"""
Interaction features: polynomial, pairwise ratios/differences, cross features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from itertools import combinations

from actions.base import BaseAction
from config.registry import register_action
from utils.validators import validate_columns_exist, validate_column_types


@register_action("interaction")
class InteractionAction(BaseAction):
    action_type = "interaction"
    domain = "feature_engineering"
    display_name = "Interaction Features"

    OPERATIONS = [
        "polynomial",
        "pairwise_ratio",
        "pairwise_difference",
        "pairwise_product",
        "cross_categorical",
    ]

    def validate(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> List[str]:
        errors = []

        operation = parameters.get("operation")
        if not operation or operation not in self.OPERATIONS:
            errors.append(
                f"Invalid operation. Must be one of {self.OPERATIONS}")
            return errors

        columns = parameters.get("columns", [])
        if not columns:
            errors.append("At least one column is required")
            return errors

        missing = validate_columns_exist(df, columns)
        if missing:
            errors.append(f"Columns not found: {missing}")

        if operation in ("polynomial", "pairwise_ratio", "pairwise_difference", "pairwise_product"):
            non_numeric = validate_column_types(df, columns, "numeric")
            if non_numeric:
                errors.append(f"Columns must be numeric: {non_numeric}")

        if operation in ("pairwise_ratio", "pairwise_difference", "pairwise_product") and len(columns) < 2:
            errors.append("Pairwise operations require at least 2 columns")

        if operation == "polynomial":
            degree = parameters.get("degree", 2)
            if not isinstance(degree, int) or degree < 2 or degree > 4:
                errors.append("'degree' must be an integer between 2 and 4")

        return errors

    def execute(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
        result = df.copy()
        operation = parameters["operation"]
        columns = parameters["columns"]

        if operation == "polynomial":
            degree = parameters.get("degree", 2)
            include_interaction = parameters.get("include_interaction", True)

            for col in columns:
                if col not in result.columns:
                    continue
                for d in range(2, degree + 1):
                    result[f"{col}_pow{d}"] = result[col] ** d

            if include_interaction and len(columns) >= 2:
                for col1, col2 in combinations(columns, 2):
                    if col1 in result.columns and col2 in result.columns:
                        result[f"{col1}_x_{col2}"] = result[col1] * \
                            result[col2]

        elif operation == "pairwise_ratio":
            for col1, col2 in combinations(columns, 2):
                if col1 in result.columns and col2 in result.columns:
                    denominator = result[col2].replace(0, np.nan)
                    result[f"{col1}_div_{col2}"] = result[col1] / denominator

        elif operation == "pairwise_difference":
            for col1, col2 in combinations(columns, 2):
                if col1 in result.columns and col2 in result.columns:
                    result[f"{col1}_minus_{col2}"] = result[col1] - \
                        result[col2]

        elif operation == "pairwise_product":
            for col1, col2 in combinations(columns, 2):
                if col1 in result.columns and col2 in result.columns:
                    result[f"{col1}_times_{col2}"] = result[col1] * \
                        result[col2]

        elif operation == "cross_categorical":
            separator = parameters.get("separator", "_")
            for col1, col2 in combinations(columns, 2):
                if col1 in result.columns and col2 in result.columns:
                    result[f"{col1}_cross_{col2}"] = (
                        result[col1].astype(str) + separator +
                        result[col2].astype(str)
                    )

        return result

    def get_parameter_schema(self) -> Dict:
        return {
            "operation": {
                "type": "string",
                "required": True,
                "choices": self.OPERATIONS,
                "description": "Interaction feature type.",
            },
            "columns": {
                "type": "list",
                "required": True,
                "description": "Columns to create interactions from.",
            },
            "degree": {
                "type": "number",
                "required": False,
                "default": 2,
                "min": 2,
                "max": 4,
                "description": "Polynomial degree.",
            },
            "include_interaction": {
                "type": "bool",
                "required": False,
                "default": True,
                "description": "Include pairwise interaction terms in polynomial.",
            },
            "separator": {
                "type": "string",
                "required": False,
                "default": "_",
                "description": "Separator for cross categorical features.",
            },
        }

    def get_description(self, parameters: Dict[str, Any]) -> str:
        operation = parameters.get("operation", "unknown")
        columns = parameters.get("columns", [])
        col_str = ", ".join(columns[:4])
        if operation == "polynomial":
            return f"Polynomial features (degree {parameters.get('degree', 2)}) for [{col_str}]"
        elif operation == "pairwise_ratio":
            return f"Pairwise ratios between [{col_str}]"
        elif operation == "pairwise_difference":
            return f"Pairwise differences between [{col_str}]"
        elif operation == "pairwise_product":
            return f"Pairwise products between [{col_str}]"
        elif operation == "cross_categorical":
            return f"Cross-categorical features from [{col_str}]"
        return f"Interaction: {operation}"

    def get_code_snippet(self, parameters: Dict[str, Any]) -> str:
        operation = parameters.get("operation")
        if operation == "polynomial":
            return "from sklearn.preprocessing import PolynomialFeatures\npoly = PolynomialFeatures(degree=2)\nX_poly = poly.fit_transform(df[columns])"
        elif operation == "pairwise_ratio":
            return "df['col1_div_col2'] = df['col1'] / df['col2']"
        return f"# Interaction: {operation}"
