"""
Numeric transformations: log, sqrt, binning, normalization, standardization, clipping.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any

from actions.base import BaseAction
from config.registry import register_action
from utils.validators import validate_columns_exist, validate_column_types


@register_action("numeric_transform")
class NumericTransformAction(BaseAction):
    action_type = "numeric_transform"
    domain = "conversion"
    display_name = "Numeric Transform"

    OPERATIONS = [
        "log", "log1p", "sqrt", "square", "power",
        "abs", "round", "clip",
        "normalize", "standardize", "robust_scale",
        "binning",
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

        non_numeric = validate_column_types(df, columns, "numeric")
        if non_numeric:
            errors.append(f"Columns must be numeric: {non_numeric}")

        operation = parameters.get("operation")
        if not operation:
            errors.append("Missing required parameter: 'operation'")
        elif operation not in self.OPERATIONS:
            errors.append(f"Invalid operation '{operation}'")

        if operation == "binning":
            n_bins = parameters.get("n_bins")
            custom_bins = parameters.get("custom_bins")
            if not n_bins and not custom_bins:
                errors.append("Binning requires 'n_bins' or 'custom_bins'")

        return errors

    def execute(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
        result = df.copy()
        columns = parameters["columns"]
        operation = parameters["operation"]
        suffix = parameters.get("suffix", "")
        overwrite = parameters.get("overwrite", True)

        for col in columns:
            if col not in result.columns:
                continue

            target_col = col if overwrite else f"{col}{suffix or '_' + operation}"
            series = result[col]

            if operation == "log":
                result[target_col] = np.log(series.clip(lower=1e-10))
            elif operation == "log1p":
                result[target_col] = np.log1p(series.clip(lower=0))
            elif operation == "sqrt":
                result[target_col] = np.sqrt(series.clip(lower=0))
            elif operation == "square":
                result[target_col] = series ** 2
            elif operation == "power":
                power = parameters.get("power", 2)
                result[target_col] = series ** power
            elif operation == "abs":
                result[target_col] = series.abs()
            elif operation == "round":
                decimals = parameters.get("decimals", 0)
                result[target_col] = series.round(decimals)
            elif operation == "clip":
                lower = parameters.get("lower")
                upper = parameters.get("upper")
                result[target_col] = series.clip(lower=lower, upper=upper)
            elif operation == "normalize":
                min_val = series.min()
                max_val = series.max()
                if max_val - min_val != 0:
                    result[target_col] = (
                        series - min_val) / (max_val - min_val)
                else:
                    result[target_col] = 0.0
            elif operation == "standardize":
                mean = series.mean()
                std = series.std()
                if std != 0:
                    result[target_col] = (series - mean) / std
                else:
                    result[target_col] = 0.0
            elif operation == "robust_scale":
                median = series.median()
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                if iqr != 0:
                    result[target_col] = (series - median) / iqr
                else:
                    result[target_col] = 0.0
            elif operation == "binning":
                n_bins = parameters.get("n_bins", 5)
                strategy = parameters.get("binning_strategy", "equal_width")
                labels = parameters.get("labels")
                custom_bins = parameters.get("custom_bins")

                bin_col = f"{col}_binned" if overwrite else target_col

                if custom_bins:
                    result[bin_col] = pd.cut(
                        series, bins=custom_bins, labels=labels, include_lowest=True)
                elif strategy == "equal_width":
                    result[bin_col] = pd.cut(
                        series, bins=n_bins, labels=labels, include_lowest=True)
                elif strategy == "equal_frequency":
                    result[bin_col] = pd.qcut(
                        series, q=n_bins, labels=labels, duplicates="drop")

        return result

    def get_parameter_schema(self) -> Dict:
        return {
            "columns": {
                "type": "list",
                "required": True,
                "description": "Numeric columns to transform.",
            },
            "operation": {
                "type": "string",
                "required": True,
                "choices": self.OPERATIONS,
                "description": "Transformation operation.",
            },
            "overwrite": {
                "type": "bool",
                "required": False,
                "default": True,
                "description": "Overwrite column or create new one.",
            },
            "suffix": {
                "type": "string",
                "required": False,
                "default": "",
                "description": "Suffix for new column name if not overwriting.",
            },
            "power": {"type": "number", "required": False, "default": 2},
            "decimals": {"type": "number", "required": False, "default": 0},
            "lower": {"type": "number", "required": False},
            "upper": {"type": "number", "required": False},
            "n_bins": {"type": "number", "required": False, "default": 5},
            "binning_strategy": {
                "type": "string",
                "required": False,
                "default": "equal_width",
                "choices": ["equal_width", "equal_frequency"],
            },
            "custom_bins": {"type": "list", "required": False},
            "labels": {"type": "list", "required": False},
        }

    def get_code_snippet(self, parameters: Dict[str, Any]) -> str:
        operation = parameters.get("operation")
        columns = parameters.get("columns", [])
        ops_map = {
            "log": f"df[{columns}] = np.log(df[{columns}].clip(lower=1e-10))",
            "log1p": f"df[{columns}] = np.log1p(df[{columns}].clip(lower=0))",
            "sqrt": f"df[{columns}] = np.sqrt(df[{columns}].clip(lower=0))",
            "square": f"df[{columns}] = df[{columns}] ** 2",
            "abs": f"df[{columns}] = df[{columns}].abs()",
            "round": f"df[{columns}] = df[{columns}].round({parameters.get('decimals', 0)})",
            "normalize": "# Min-Max normalization\ndf[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())",
            "standardize": "# Z-score standardization\ndf[col] = (df[col] - df[col].mean()) / df[col].std()",
            "robust_scale": "# Robust scaling\ndf[col] = (df[col] - df[col].median()) / (df[col].quantile(0.75) - df[col].quantile(0.25))",
            "binning": f"df['{columns[0]}_binned'] = pd.cut(df['{columns[0]}'], bins={parameters.get('n_bins', 5)})",
        }

        return ops_map.get(
            operation, f"# Numeric transform: {operation}"  # type: ignore
        )
