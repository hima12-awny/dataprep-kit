"""
Temporal features: lag, lead, rolling windows, cyclical encoding.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any

from actions.base import BaseAction
from config.registry import register_action
from utils.validators import validate_columns_exist


@register_action("temporal")
class TemporalAction(BaseAction):
    action_type = "temporal"
    domain = "feature_engineering"
    display_name = "Temporal Features"

    OPERATIONS = [
        "lag",
        "lead",
        "rolling",
        "cyclical_encoding",
    ]

    ROLLING_FUNCTIONS = ["mean", "sum", "std", "min", "max", "median"]

    def validate(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> List[str]:
        errors = []

        operation = parameters.get("operation")
        if not operation or operation not in self.OPERATIONS:
            errors.append(f"Invalid operation. Must be one of {self.OPERATIONS}")
            return errors

        if operation in ("lag", "lead", "rolling"):
            columns = parameters.get("columns", [])
            if not columns:
                errors.append("'columns' parameter is required")
            missing = validate_columns_exist(df, columns)
            if missing:
                errors.append(f"Columns not found: {missing}")

        if operation in ("lag", "lead"):
            periods = parameters.get("periods", 1)
            if not isinstance(periods, int) or periods < 1:
                errors.append("'periods' must be a positive integer")

        if operation == "rolling":
            window = parameters.get("window")
            if not window or not isinstance(window, int) or window < 2:
                errors.append("'window' must be an integer >= 2")
            func = parameters.get("rolling_func", "mean")
            if func not in self.ROLLING_FUNCTIONS:
                errors.append(f"Invalid rolling_func. Must be one of {self.ROLLING_FUNCTIONS}")

        if operation == "cyclical_encoding":
            column = parameters.get("column")
            if not column:
                errors.append("'column' is required for cyclical encoding")
            elif column not in df.columns:
                errors.append(f"Column '{column}' not found")
            max_value = parameters.get("max_value")
            if max_value is None:
                errors.append("'max_value' is required (e.g., 24 for hours, 7 for weekdays)")

        return errors

    def execute(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
        result = df.copy()
        operation = parameters["operation"]

        if operation == "lag":
            columns = parameters["columns"]
            periods = parameters.get("periods", 1)
            group_by = parameters.get("group_by")

            for col in columns:
                if col not in result.columns:
                    continue
                new_col = f"{col}_lag_{periods}"
                if group_by:
                    result[new_col] = result.groupby(group_by)[col].shift(periods)
                else:
                    result[new_col] = result[col].shift(periods)

        elif operation == "lead":
            columns = parameters["columns"]
            periods = parameters.get("periods", 1)
            group_by = parameters.get("group_by")

            for col in columns:
                if col not in result.columns:
                    continue
                new_col = f"{col}_lead_{periods}"
                if group_by:
                    result[new_col] = result.groupby(group_by)[col].shift(-periods)
                else:
                    result[new_col] = result[col].shift(-periods)

        elif operation == "rolling":
            columns = parameters["columns"]
            window = parameters["window"]
            func = parameters.get("rolling_func", "mean")
            group_by = parameters.get("group_by")
            min_periods = parameters.get("min_periods", 1)

            for col in columns:
                if col not in result.columns:
                    continue
                new_col = f"{col}_rolling_{func}_{window}"

                if group_by:
                    roller = result.groupby(group_by)[col].rolling(
                        window=window, min_periods=min_periods
                    )
                    result[new_col] = getattr(roller, func)().reset_index(level=0, drop=True)
                else:
                    roller = result[col].rolling(window=window, min_periods=min_periods)
                    result[new_col] = getattr(roller, func)()

        elif operation == "cyclical_encoding":
            column = parameters["column"]
            max_value = parameters["max_value"]

            values = result[column].astype(float)
            result[f"{column}_sin"] = np.sin(2 * np.pi * values / max_value)
            result[f"{column}_cos"] = np.cos(2 * np.pi * values / max_value)

            if parameters.get("drop_original", False):
                result = result.drop(columns=[column])

        return result

    def get_parameter_schema(self) -> Dict:
        return {
            "operation": {
                "type": "string",
                "required": True,
                "choices": self.OPERATIONS,
                "description": "Temporal feature operation.",
            },
            "columns": {
                "type": "list",
                "required": False,
                "description": "Columns for lag/lead/rolling.",
            },
            "column": {
                "type": "string",
                "required": False,
                "description": "Column for cyclical encoding.",
            },
            "periods": {
                "type": "number",
                "required": False,
                "default": 1,
                "description": "Number of periods for lag/lead.",
            },
            "window": {
                "type": "number",
                "required": False,
                "description": "Rolling window size.",
            },
            "rolling_func": {
                "type": "string",
                "required": False,
                "default": "mean",
                "choices": self.ROLLING_FUNCTIONS,
                "description": "Rolling aggregation function.",
            },
            "group_by": {
                "type": "string",
                "required": False,
                "description": "Group column for grouped operations.",
            },
            "min_periods": {
                "type": "number",
                "required": False,
                "default": 1,
                "description": "Minimum periods for rolling.",
            },
            "max_value": {
                "type": "number",
                "required": False,
                "description": "Max cycle value for cyclical encoding (e.g., 24 for hours).",
            },
            "drop_original": {
                "type": "bool",
                "required": False,
                "default": False,
                "description": "Drop original column after cyclical encoding.",
            },
        }

    def get_description(self, parameters: Dict[str, Any]) -> str:
        operation = parameters.get("operation", "unknown")
        if operation == "lag":
            return f"Create lag({parameters.get('periods', 1)}) features for {parameters.get('columns')}"
        elif operation == "lead":
            return f"Create lead({parameters.get('periods', 1)}) features for {parameters.get('columns')}"
        elif operation == "rolling":
            return f"Rolling {parameters.get('rolling_func', 'mean')}(window={parameters.get('window')}) for {parameters.get('columns')}"
        elif operation == "cyclical_encoding":
            return f"Cyclical (sin/cos) encoding of '{parameters.get('column')}' (max={parameters.get('max_value')})"
        return f"Temporal feature: {operation}"

    def get_code_snippet(self, parameters: Dict[str, Any]) -> str:
        operation = parameters.get("operation")
        if operation == "lag":
            return f"df['col_lag'] = df['col'].shift({parameters.get('periods', 1)})"
        elif operation == "rolling":
            return f"df['col_rolling'] = df['col'].rolling({parameters.get('window')}).{parameters.get('rolling_func', 'mean')}()"
        elif operation == "cyclical_encoding":
            return f"df['col_sin'] = np.sin(2 * np.pi * df['col'] / {parameters.get('max_value')})\ndf['col_cos'] = np.cos(2 * np.pi * df['col'] / {parameters.get('max_value')})"
        return f"# Temporal: {operation}"