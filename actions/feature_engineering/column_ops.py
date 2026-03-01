"""
Column operations: create from expression, combine, split, rename, drop, reorder.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any

from actions.base import BaseAction
from config.registry import register_action
from utils.validators import validate_columns_exist


@register_action("column_ops")
class ColumnOpsAction(BaseAction):
    action_type = "column_ops"
    domain = "feature_engineering"
    display_name = "Column Operations"

    OPERATIONS = [
        "create_expression",
        "combine_columns",
        "split_column",
        "rename_columns",
        "drop_columns",
        "reorder_columns",
    ]

    def validate(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> List[str]:
        errors = []

        operation = parameters.get("operation")
        if not operation:
            errors.append("Missing required parameter: 'operation'")
            return errors

        if operation not in self.OPERATIONS:
            errors.append(f"Invalid operation '{operation}'")
            return errors

        if operation == "create_expression":
            if not parameters.get("new_column"):
                errors.append("'create_expression' requires 'new_column' name")
            if not parameters.get("expression"):
                errors.append("'create_expression' requires 'expression'")

        elif operation == "combine_columns":
            columns = parameters.get("columns", [])
            if len(columns) < 2:
                errors.append("'combine_columns' requires at least 2 columns")
            missing = validate_columns_exist(df, columns)
            if missing:
                errors.append(f"Columns not found: {missing}")

        elif operation == "split_column":
            column = parameters.get("column")
            if not column:
                errors.append("'split_column' requires 'column'")
            elif column not in df.columns:
                errors.append(f"Column '{column}' not found")
            if not parameters.get("delimiter"):
                errors.append("'split_column' requires 'delimiter'")

        elif operation == "rename_columns":
            rename_map = parameters.get("rename_map", {})
            if not rename_map:
                errors.append("'rename_columns' requires 'rename_map' dict")

        elif operation == "drop_columns":
            columns = parameters.get("columns", [])
            if not columns:
                errors.append("'drop_columns' requires 'columns' list")
            missing = validate_columns_exist(df, columns)
            if missing:
                errors.append(f"Columns not found: {missing}")

        elif operation == "reorder_columns":
            new_order = parameters.get("new_order", [])
            if not new_order:
                errors.append("'reorder_columns' requires 'new_order' list")

        return errors

    def execute(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
        result = df.copy()
        operation = parameters["operation"]

        if operation == "create_expression":
            new_column = parameters["new_column"]
            expression = parameters["expression"]
            try:
                # Safe evaluation using dataframe columns
                # Users can reference columns like: col_A + col_B, col_A / col_B
                local_vars = {col: result[col] for col in result.columns}
                local_vars["np"] = np
                local_vars["pd"] = pd
                result[new_column] = eval(
                    expression, {"__builtins__": {}}, local_vars)
            except Exception as e:
                raise ValueError(f"Expression evaluation failed: {e}")

        elif operation == "combine_columns":
            columns = parameters["columns"]
            new_column = parameters.get("new_column", "_".join(columns))
            separator = parameters.get("separator", "_")
            result[new_column] = result[columns].astype(
                str).agg(separator.join, axis=1)

        elif operation == "split_column":
            column = parameters["column"]
            delimiter = parameters["delimiter"]
            max_splits = parameters.get("max_splits", -1)
            prefix = parameters.get("prefix", column)

            splits = result[column].astype(str).str.split(
                delimiter, n=max_splits if max_splits > 0 else None, expand=True)
            for i in range(splits.shape[1]):
                result[f"{prefix}_{i}"] = splits[i]

            if parameters.get("drop_original", False):
                result = result.drop(columns=[column])

        elif operation == "rename_columns":
            rename_map = parameters["rename_map"]
            result = result.rename(columns=rename_map)

        elif operation == "drop_columns":
            columns = parameters["columns"]
            result = result.drop(
                columns=[c for c in columns if c in result.columns])

        elif operation == "reorder_columns":
            new_order = parameters["new_order"]
            remaining = [c for c in result.columns if c not in new_order]
            result = result[new_order + remaining]

        return result  # type: ignore

    def get_parameter_schema(self) -> Dict:
        return {
            "operation": {
                "type": "string",
                "required": True,
                "choices": self.OPERATIONS,
                "description": "Column operation to perform.",
            },
            "columns": {
                "type": "list",
                "required": False,
                "description": "Columns for combine/drop operations.",
            },
            "column": {
                "type": "string",
                "required": False,
                "description": "Single column for split operation.",
            },
            "new_column": {
                "type": "string",
                "required": False,
                "description": "Name for the newly created column.",
            },
            "expression": {
                "type": "string",
                "required": False,
                "description": "Python expression for column creation (e.g., 'col_A / col_B').",
            },
            "separator": {
                "type": "string",
                "required": False,
                "default": "_",
                "description": "Separator for combining columns.",
            },
            "delimiter": {
                "type": "string",
                "required": False,
                "description": "Delimiter for splitting a column.",
            },
            "max_splits": {
                "type": "number",
                "required": False,
                "default": -1,
                "description": "Max number of splits (-1 for all).",
            },
            "prefix": {
                "type": "string",
                "required": False,
                "description": "Prefix for split result columns.",
            },
            "rename_map": {
                "type": "any",
                "required": False,
                "description": "Dict mapping old column names to new names.",
            },
            "new_order": {
                "type": "list",
                "required": False,
                "description": "New column order.",
            },
            "drop_original": {
                "type": "bool",
                "required": False,
                "default": False,
                "description": "Drop the original column after split.",
            },
        }

    def get_description(self, parameters: Dict[str, Any]) -> str:
        operation = parameters.get("operation", "unknown")
        if operation == "create_expression":
            return f"Create column '{parameters.get('new_column')}' = {parameters.get('expression')}"
        elif operation == "combine_columns":
            return f"Combine {parameters.get('columns')} into '{parameters.get('new_column')}'"
        elif operation == "split_column":
            return f"Split '{parameters.get('column')}' by '{parameters.get('delimiter')}'"
        elif operation == "rename_columns":
            return f"Rename {len(parameters.get('rename_map', {}))} columns"
        elif operation == "drop_columns":
            return f"Drop columns: {parameters.get('columns')}"
        elif operation == "reorder_columns":
            return "Reorder columns"
        return f"Column operation: {operation}"

    def get_code_snippet(self, parameters: Dict[str, Any]) -> str:
        operation = parameters.get("operation")
        if operation == "create_expression":
            return f"df['{parameters.get('new_column')}'] = {parameters.get('expression')}"
        elif operation == "drop_columns":
            return f"df = df.drop(columns={parameters.get('columns')})"
        elif operation == "rename_columns":
            return f"df = df.rename(columns={parameters.get('rename_map')})"
        return f"# Column operation: {operation}"
