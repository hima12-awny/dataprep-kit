"""
Categorical encoding: label, one-hot, target, frequency, binary, ordinal.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any

from actions.base import BaseAction
from config.registry import register_action
from utils.validators import validate_columns_exist


@register_action("encoding")
class EncodingAction(BaseAction):
    action_type = "encoding"
    domain = "conversion"
    display_name = "Categorical Encoding"

    METHODS = [
        "label", "onehot", "frequency",
        "ordinal", "binary", "target",
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

        method = parameters.get("method")
        if not method:
            errors.append("Missing required parameter: 'method'")
        elif method not in self.METHODS:
            errors.append(
                f"Invalid method '{method}'. Must be one of {self.METHODS}")

        if method == "ordinal":
            order = parameters.get("order")
            if not order or not isinstance(order, dict):
                errors.append(
                    "'ordinal' encoding requires 'order' dict: {column: [ordered_values]}")

        if method == "target":
            target_col = parameters.get("target_column")
            if not target_col:
                errors.append(
                    "'target' encoding requires 'target_column' parameter")
            elif target_col not in df.columns:
                errors.append(f"Target column '{target_col}' not found")

        return errors

    def execute(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
        result = df.copy()
        columns = parameters["columns"]
        method = parameters["method"]

        if method == "label":
            for col in columns:
                if col in result.columns:
                    categories = result[col].astype(str).unique()
                    cat_map = {cat: idx for idx,
                               cat in enumerate(sorted(categories))}
                    result[col] = result[col].astype(str)\
                        .map(cat_map)  # type: ignore

        elif method == "onehot":
            prefix = parameters.get("prefix", None)
            drop_first = parameters.get("drop_first", False)
            max_categories = parameters.get("max_categories")

            for col in columns:
                if col not in result.columns:
                    continue

                col_prefix = prefix if prefix else col

                if max_categories:
                    top_cats = result[col].value_counts().head(
                        max_categories).index
                    result[col] = result[col].where(
                        result[col].isin(top_cats), other="__other__")

                dummies = pd.get_dummies(
                    result[col],
                    prefix=col_prefix,
                    drop_first=drop_first,
                    dtype=int,
                )
                result = pd.concat([result, dummies], axis=1)

            drop_original = parameters.get("drop_original", True)
            if drop_original:
                result = result.drop(
                    columns=[c for c in columns if c in result.columns])

        elif method == "frequency":
            for col in columns:
                if col in result.columns:
                    freq_map = result[col].value_counts(
                        normalize=True).to_dict()
                    new_col = parameters.get("suffix", "_freq_encoded")
                    result[f"{col}{new_col}"] = result[col]\
                        .map(freq_map)  # type: ignore

        elif method == "ordinal":
            order = parameters["order"]
            for col in columns:
                if col in result.columns and col in order:
                    cat_order = order[col]
                    ordinal_map = {val: idx for idx,
                                   val in enumerate(cat_order)}
                    result[col] = result[col]\
                        .map(ordinal_map)  # type: ignore

        elif method == "binary":
            for col in columns:
                if col in result.columns:
                    categories = result[col].astype(str).unique()
                    cat_map = {cat: idx for idx,
                               cat in enumerate(sorted(categories))}
                    encoded = result[col].astype(str)\
                        .map(cat_map)  # type: ignore

                    max_val = max(cat_map.values()) if cat_map else 0
                    n_bits = max(1, int(np.ceil(np.log2(max_val + 1)))
                                 ) if max_val > 0 else 1

                    for bit in range(n_bits):
                        result[f"{col}_bit_{bit}"] = (
                            (encoded // (2 ** bit)) % 2).astype(int)

                    drop_original = parameters.get("drop_original", True)
                    if drop_original and col in result.columns:
                        result = result.drop(columns=[col])

        elif method == "target":
            target_col = parameters["target_column"]
            smoothing = parameters.get("smoothing", 1.0)
            global_mean = result[target_col].mean()

            for col in columns:
                if col not in result.columns:
                    continue
                stats = result.groupby(col)[target_col].agg(["mean", "count"])
                stats["smoothed"] = (  # type: ignore
                    (stats["count"] * stats["mean"] + smoothing * global_mean)
                    / (stats["count"] + smoothing)
                )
                new_col = f"{col}_target_encoded"
                result[new_col] = result[col].map(
                    stats["smoothed"])  # type: ignore

        return result

    def get_parameter_schema(self) -> Dict:
        return {
            "columns": {
                "type": "list",
                "required": True,
                "description": "Categorical columns to encode.",
            },
            "method": {
                "type": "string",
                "required": True,
                "choices": self.METHODS,
                "description": "Encoding method.",
            },
            "drop_first": {
                "type": "bool",
                "required": False,
                "default": False,
                "description": "Drop first dummy column (one-hot).",
            },
            "drop_original": {
                "type": "bool",
                "required": False,
                "default": True,
                "description": "Drop original column after encoding.",
            },
            "prefix": {
                "type": "string",
                "required": False,
                "description": "Prefix for one-hot encoded columns.",
            },
            "max_categories": {
                "type": "number",
                "required": False,
                "description": "Max categories for one-hot (rest grouped as 'other').",
            },
            "order": {
                "type": "any",
                "required": False,
                "description": "Dict of {column: [ordered_values]} for ordinal encoding.",
            },
            "target_column": {
                "type": "string",
                "required": False,
                "description": "Target column for target encoding.",
            },
            "smoothing": {
                "type": "number",
                "required": False,
                "default": 1.0,
                "description": "Smoothing factor for target encoding.",
            },
            "suffix": {
                "type": "string",
                "required": False,
                "default": "_freq_encoded",
                "description": "Suffix for frequency encoding new column.",
            },
        }

    def get_description(self, parameters: Dict[str, Any]) -> str:
        method = parameters.get("method", "unknown")
        columns = parameters.get("columns", [])
        col_str = ", ".join(columns[:3])
        if len(columns) > 3:
            col_str += f" (+{len(columns)-3} more)"
        return f"Apply '{method}' encoding to [{col_str}]"

    def get_code_snippet(self, parameters: Dict[str, Any]) -> str:
        method = parameters.get("method")
        columns = parameters.get("columns", [])
        snippets = {
            "label": f"from sklearn.preprocessing import LabelEncoder\nle = LabelEncoder()\ndf[{columns}] = df[{columns}].apply(le.fit_transform)",
            "onehot": f"df = pd.get_dummies(df, columns={columns}, drop_first={parameters.get('drop_first', False)})",
            "frequency": f"for col in {columns}:\n    df[col+'_freq'] = df[col].map(df[col].value_counts(normalize=True))",
            "ordinal": f"# Ordinal encoding with custom order\n# order = {parameters.get('order', {})}",
            "target": f"# Target encoding with smoothing={parameters.get('smoothing', 1.0)}",
            "binary": f"# Binary encoding for {columns}",
        }
        return snippets.get(method, f"# Encoding: {method}")  # type: ignore
