"""
Detect and handle outliers using IQR, Z-score, or percentile methods.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from scipy import stats as scipy_stats

from actions.base import BaseAction
from config.registry import register_action
from utils.validators import validate_columns_exist, validate_column_types


@register_action("handle_outliers")
class HandleOutliersAction(BaseAction):
    action_type = "handle_outliers"
    domain = "cleaning"
    display_name = "Handle Outliers"

    METHODS = ["iqr", "zscore", "percentile"]
    BEHAVIORS = ["remove", "clip", "flag"]

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

        method = parameters.get("method", "iqr")
        if method not in self.METHODS:
            errors.append(
                f"Invalid method '{method}'. Must be one of {self.METHODS}")

        behavior = parameters.get("behavior", "clip")
        if behavior not in self.BEHAVIORS:
            errors.append(
                f"Invalid behavior '{behavior}'. Must be one of {self.BEHAVIORS}")

        threshold = parameters.get("threshold")
        if method == "iqr" and threshold is not None and threshold <= 0:
            errors.append("IQR threshold must be positive")
        if method == "zscore" and threshold is not None and threshold <= 0:
            errors.append("Z-score threshold must be positive")

        return errors

    def execute(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
        result = df.copy()
        columns = parameters["columns"]
        method = parameters.get("method", "iqr")
        behavior = parameters.get("behavior", "clip")
        threshold = parameters.get("threshold")

        for col in columns:
            if col not in result.columns:  # type: ignore
                continue

            series = result[col]
            lower, upper = self._compute_bounds(
                series,  # type: ignore
                method,
                threshold,
                parameters)

            if lower is None or upper is None:
                continue

            outlier_mask = (series < lower) | (series > upper)

            if behavior == "remove":
                result = result[~outlier_mask]

            elif behavior == "clip":
                result[col] = series.clip(
                    lower=lower, upper=upper)  # type: ignore

            elif behavior == "flag":
                flag_col = f"{col}_outlier_flag"
                result[flag_col] = outlier_mask.astype(int)

        if behavior == "remove":
            result = result.reset_index(drop=True)  # type: ignore

        return result  # type: ignore

    def _compute_bounds(
        self, series: pd.Series, method: str, threshold: Any, parameters: Dict
    ):
        """Compute lower and upper bounds for outlier detection."""
        non_null = series.dropna()
        if len(non_null) < 3:
            return None, None

        if method == "iqr":
            multiplier = threshold if threshold else 1.5
            q1 = non_null.quantile(0.25)
            q3 = non_null.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - multiplier * iqr
            upper = q3 + multiplier * iqr

        elif method == "zscore":
            z_thresh = threshold if threshold else 3.0
            mean = non_null.mean()
            std = non_null.std()
            if std == 0:
                return None, None
            lower = mean - z_thresh * std
            upper = mean + z_thresh * std

        elif method == "percentile":
            lower_pct = parameters.get("lower_percentile", 0.01)
            upper_pct = parameters.get("upper_percentile", 0.99)
            lower = non_null.quantile(lower_pct)
            upper = non_null.quantile(upper_pct)

        else:
            return None, None

        return lower, upper

    def preview(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> Dict:
        base_preview = super().preview(df, parameters)

        # Add outlier-specific info
        columns = parameters.get("columns", [])
        method = parameters.get("method", "iqr")
        threshold = parameters.get("threshold")
        outlier_info = {}

        for col in columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                lower, upper = self._compute_bounds(
                    df[col],  # type: ignore
                    method,
                    threshold,
                    parameters
                )
                if lower is not None:
                    mask = (df[col] < lower) | (df[col] > upper)
                    outlier_info[col] = {
                        "count": int(mask.sum()),
                        "percentage": round(mask.mean() * 100, 2),
                        "lower_bound": round(float(lower), 4),
                        "upper_bound": round(float(upper), 4),  # type: ignore
                    }

        base_preview["outlier_details"] = outlier_info
        return base_preview

    def get_parameter_schema(self) -> Dict:
        return {
            "columns": {
                "type": "list",
                "required": True,
                "description": "Numeric columns to check for outliers.",
            },
            "method": {
                "type": "string",
                "required": False,
                "default": "iqr",
                "choices": self.METHODS,
                "description": "Outlier detection method.",
            },
            "threshold": {
                "type": "number",
                "required": False,
                "default": None,
                "description": "IQR multiplier (default 1.5) or Z-score threshold (default 3.0).",
            },
            "behavior": {
                "type": "string",
                "required": False,
                "default": "clip",
                "choices": self.BEHAVIORS,
                "description": "How to handle detected outliers.",
            },
            "lower_percentile": {
                "type": "number",
                "required": False,
                "default": 0.01,
                "min": 0,
                "max": 0.5,
                "description": "Lower percentile for 'percentile' method.",
            },
            "upper_percentile": {
                "type": "number",
                "required": False,
                "default": 0.99,
                "min": 0.5,
                "max": 1,
                "description": "Upper percentile for 'percentile' method.",
            },
        }

    def get_description(self, parameters: Dict[str, Any]) -> str:
        columns = parameters.get("columns", [])
        method = parameters.get("method", "iqr")
        behavior = parameters.get("behavior", "clip")
        col_str = ", ".join(columns[:3])
        if len(columns) > 3:
            col_str += f" (+{len(columns)-3} more)"
        return f"Handle outliers in [{col_str}] using {method.upper()} method → {behavior}"

    def get_code_snippet(self, parameters: Dict[str, Any]) -> str:
        method = parameters.get("method", "iqr")
        behavior = parameters.get("behavior", "clip")
        columns = parameters.get("columns", [])

        if method == "iqr" and behavior == "clip":
            return (
                f"for col in {columns}:\n"
                f"    Q1 = df[col].quantile(0.25)\n"
                f"    Q3 = df[col].quantile(0.75)\n"
                f"    IQR = Q3 - Q1\n"
                f"    df[col] = df[col].clip(lower=Q1 - 1.5*IQR, upper=Q3 + 1.5*IQR)"
            )
        return f"# Handle outliers: method={method}, behavior={behavior}"
