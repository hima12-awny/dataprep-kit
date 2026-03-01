"""
Handle missing values: drop, impute with various strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

from actions.base import BaseAction
from config.registry import register_action
from utils.validators import validate_columns_exist


@register_action("handle_missing")
class HandleMissingAction(BaseAction):
    action_type = "handle_missing"
    domain = "cleaning"
    display_name = "Handle Missing Values"
    description_template = "Handle missing values in {columns} using {strategy} strategy"

    STRATEGIES = [
        "mean", "median", "mode", "constant",
        "forward_fill", "backward_fill",
        "drop_rows", "drop_columns", "group_based",
    ]

    def validate(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> List[str]:
        errors = []

        strategy = parameters.get("strategy")
        if not strategy:
            errors.append("Missing required parameter: 'strategy'")
            return errors

        if strategy not in self.STRATEGIES:
            errors.append(f"Invalid strategy '{strategy}'. Must be one of {self.STRATEGIES}")
            return errors

        columns = parameters.get("columns")
        if strategy != "drop_columns" and columns:
            missing = validate_columns_exist(df, columns)
            if missing:
                errors.append(f"Columns not found: {missing}")

        # Validate numeric-only strategies
        numeric_only = {"mean", "median"}
        if strategy in numeric_only and columns:
            for col in columns:
                if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                    errors.append(f"Strategy '{strategy}' requires numeric column, but '{col}' is {df[col].dtype}")

        # Validate constant fill
        if strategy == "constant" and parameters.get("fill_value") is None:
            errors.append("Strategy 'constant' requires 'fill_value' parameter")

        # Validate group_based
        if strategy == "group_based":
            if not parameters.get("group_by"):
                errors.append("Strategy 'group_based' requires 'group_by' parameter")
            if not parameters.get("group_strategy"):
                errors.append("Strategy 'group_based' requires 'group_strategy' parameter")

        return errors

    def execute(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
        result = df.copy()
        strategy = parameters["strategy"]
        columns = parameters.get("columns") or result.columns.tolist()

        # Filter to only columns that exist
        columns = [c for c in columns if c in result.columns]

        if strategy == "drop_rows":
            how = parameters.get("how", "any")  # "any" or "all"
            threshold = parameters.get("threshold")
            if threshold is not None:
                result = result.dropna(subset=columns, thresh=threshold)
            else:
                result = result.dropna(subset=columns, how=how)

        elif strategy == "drop_columns":
            max_missing_pct = parameters.get("threshold", 0.5)
            cols_to_drop = []
            for col in columns:
                if col in result.columns:
                    pct = result[col].isna().mean()
                    if pct > max_missing_pct:
                        cols_to_drop.append(col)
            result = result.drop(columns=cols_to_drop)

        elif strategy == "mean":
            for col in columns:
                if pd.api.types.is_numeric_dtype(result[col]):
                    result[col] = result[col].fillna(result[col].mean())

        elif strategy == "median":
            for col in columns:
                if pd.api.types.is_numeric_dtype(result[col]):
                    result[col] = result[col].fillna(result[col].median())

        elif strategy == "mode":
            for col in columns:
                mode_val = result[col].mode()
                if len(mode_val) > 0:
                    result[col] = result[col].fillna(mode_val.iloc[0])

        elif strategy == "constant":
            fill_value = parameters["fill_value"]
            for col in columns:
                result[col] = result[col].fillna(fill_value)

        elif strategy == "forward_fill":
            for col in columns:
                result[col] = result[col].ffill()

        elif strategy == "backward_fill":
            for col in columns:
                result[col] = result[col].bfill()

        elif strategy == "group_based":
            group_by = parameters["group_by"]
            group_strategy = parameters.get("group_strategy", "mean")
            for col in columns:
                if col == group_by:
                    continue
                if group_strategy == "mean" and pd.api.types.is_numeric_dtype(result[col]):
                    result[col] = result.groupby(group_by)[col].transform(
                        lambda x: x.fillna(x.mean())
                    )
                elif group_strategy == "median" and pd.api.types.is_numeric_dtype(result[col]):
                    result[col] = result.groupby(group_by)[col].transform(
                        lambda x: x.fillna(x.median())
                    )
                elif group_strategy == "mode":
                    result[col] = result.groupby(group_by)[col].transform(
                        lambda x: x.fillna(x.mode().iloc[0] if len(x.mode()) > 0 else x)
                    )

        return result

    def get_parameter_schema(self) -> Dict:
        return {
            "columns": {
                "type": "list",
                "required": False,
                "default": None,
                "description": "Columns to process. None = all columns.",
            },
            "strategy": {
                "type": "string",
                "required": True,
                "choices": self.STRATEGIES,
                "description": "Imputation or removal strategy.",
            },
            "fill_value": {
                "type": "any",
                "required": False,
                "default": None,
                "description": "Value for 'constant' strategy.",
            },
            "how": {
                "type": "string",
                "required": False,
                "default": "any",
                "choices": ["any", "all"],
                "description": "For 'drop_rows': drop if 'any' or 'all' values are null.",
            },
            "threshold": {
                "type": "number",
                "required": False,
                "default": None,
                "description": "For 'drop_columns': max missing percentage (0-1). For 'drop_rows': min non-null count.",
            },
            "group_by": {
                "type": "string",
                "required": False,
                "default": None,
                "description": "Column to group by for 'group_based' strategy.",
            },
            "group_strategy": {
                "type": "string",
                "required": False,
                "default": "mean",
                "choices": ["mean", "median", "mode"],
                "description": "Aggregation used within groups.",
            },
            "create_indicator": {
                "type": "bool",
                "required": False,
                "default": False,
                "description": "Create binary indicator column before imputing.",
            },
        }

    def get_description(self, parameters: Dict[str, Any]) -> str:
        strategy = parameters.get("strategy", "unknown")
        columns = parameters.get("columns", "all columns")
        if isinstance(columns, list):
            if len(columns) <= 3:
                columns = ", ".join(columns)
            else:
                columns = f"{len(columns)} columns"
        return f"Handle missing values in {columns} using '{strategy}' strategy"

    def get_code_snippet(self, parameters: Dict[str, Any]) -> str:
        strategy = parameters.get("strategy")
        columns = parameters.get("columns")
        col_str = f"[{', '.join(repr(c) for c in columns)}]" if columns else "df.columns"

        snippets = {
            "mean": f"df[{col_str}] = df[{col_str}].fillna(df[{col_str}].mean())",
            "median": f"df[{col_str}] = df[{col_str}].fillna(df[{col_str}].median())",
            "mode": f"for col in {col_str}:\n    df[col] = df[col].fillna(df[col].mode().iloc[0])",
            "constant": f"df[{col_str}] = df[{col_str}].fillna({repr(parameters.get('fill_value'))})",
            "forward_fill": f"df[{col_str}] = df[{col_str}].ffill()",
            "backward_fill": f"df[{col_str}] = df[{col_str}].bfill()",
            "drop_rows": f"df = df.dropna(subset={col_str})",
            "drop_columns": f"# Drop columns with >{parameters.get('threshold', 0.5)*100}% missing",
        }
        return snippets.get(strategy, f"# {strategy} strategy") # type: ignore