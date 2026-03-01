"""
Aggregation features: group-by, pivot, melt, cumulative operations.
"""

import pandas as pd
from typing import Dict, List, Any

from actions.base import BaseAction
from config.registry import register_action
from utils.validators import validate_columns_exist


@register_action("aggregation")
class AggregationAction(BaseAction):
    action_type = "aggregation"
    domain = "feature_engineering"
    display_name = "Aggregation"

    OPERATIONS = [
        "group_aggregate",
        "pivot",
        "melt",
        "cumulative",
    ]

    AGG_FUNCTIONS = ["sum", "mean", "median", "min",
                     "max", "count", "std", "var", "first", "last"]

    CUMULATIVE_FUNCTIONS = ["cumsum", "cumcount", "cumpct"]

    def validate(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> List[str]:
        errors = []

        operation = parameters.get("operation")
        if not operation:
            errors.append("Missing required parameter: 'operation'")
            return errors

        if operation not in self.OPERATIONS:
            errors.append(f"Invalid operation '{operation}'")

        if operation == "group_aggregate":
            group_by = parameters.get("group_by", [])
            if not group_by:
                errors.append("'group_aggregate' requires 'group_by' columns")
            agg_columns = parameters.get("agg_columns", [])
            if not agg_columns:
                errors.append("'group_aggregate' requires 'agg_columns'")
            agg_func = parameters.get("agg_func", "mean")
            if agg_func not in self.AGG_FUNCTIONS:
                errors.append(f"Invalid agg_func '{agg_func}'")

            all_cols = group_by + agg_columns
            missing = validate_columns_exist(df, all_cols)
            if missing:
                errors.append(f"Columns not found: {missing}")

        elif operation == "pivot":
            for key in ["index", "pivot_column", "value_column"]:
                if not parameters.get(key):
                    errors.append(f"'pivot' requires '{key}' parameter")

        elif operation == "melt":
            if not parameters.get("id_vars"):
                errors.append("'melt' requires 'id_vars' parameter")

        elif operation == "cumulative":
            if not parameters.get("columns"):
                errors.append("'cumulative' requires 'columns'")
            cum_func = parameters.get("cum_func", "cumsum")
            if cum_func not in self.CUMULATIVE_FUNCTIONS:
                errors.append(f"Invalid cum_func '{cum_func}'")

        return errors

    def execute(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
        result = df.copy()
        operation = parameters["operation"]

        if operation == "group_aggregate":
            group_by = parameters["group_by"]
            agg_columns = parameters["agg_columns"]
            agg_func = parameters.get("agg_func", "mean")
            merge_back = parameters.get("merge_back", True)

            agg_result = result.groupby(
                group_by)[agg_columns].agg(agg_func).reset_index()

            if merge_back:
                suffix = f"_{agg_func}"
                rename_map = {col: f"{col}{suffix}" for col in agg_columns}
                agg_result = agg_result.rename(columns=rename_map)
                result = result.merge(agg_result, on=group_by, how="left")
            else:
                result = agg_result

        elif operation == "pivot":
            index = parameters["index"]
            pivot_column = parameters["pivot_column"]
            value_column = parameters["value_column"]
            agg_func = parameters.get("agg_func", "mean")

            result = pd.pivot_table(
                result,
                index=index,
                columns=pivot_column,
                values=value_column,
                aggfunc=agg_func,
            ).reset_index()
            # Flatten multi-level columns
            if isinstance(result.columns, pd.MultiIndex):
                result.columns = [
                    "_".join(str(c) for c in col).strip("_")
                    for col in result.columns.values
                ]

        elif operation == "melt":
            id_vars = parameters["id_vars"]
            value_vars = parameters.get("value_vars")
            var_name = parameters.get("var_name", "variable")
            value_name = parameters.get("value_name", "value")

            result = result.melt(
                id_vars=id_vars,
                value_vars=value_vars,
                var_name=var_name,
                value_name=value_name,
            )

        elif operation == "cumulative":
            columns = parameters["columns"]
            cum_func = parameters.get("cum_func", "cumsum")
            group_by = parameters.get("group_by")

            for col in columns:
                if col not in result.columns:
                    continue

                new_col = f"{col}_{cum_func}"

                if group_by:
                    grouped = result.groupby(group_by)[col]
                else:
                    grouped = result[col]

                if cum_func == "cumsum":
                    result[new_col] = grouped.cumsum(
                    ) if group_by is None else grouped.transform("cumsum")
                elif cum_func == "cumcount":
                    if group_by:
                        result[new_col] = result.groupby(group_by).cumcount()
                    else:
                        result[new_col] = range(len(result))
                elif cum_func == "cumpct":
                    total = grouped.transform(
                        "sum") if group_by else result[col].sum()
                    cum_val = grouped.transform(
                        "cumsum") if group_by else result[col].cumsum()
                    result[new_col] = cum_val / total

        return result

    def get_parameter_schema(self) -> Dict:
        return {
            "operation": {
                "type": "string",
                "required": True,
                "choices": self.OPERATIONS,
                "description": "Aggregation operation.",
            },
            "group_by": {
                "type": "list",
                "required": False,
                "description": "Columns to group by.",
            },
            "agg_columns": {
                "type": "list",
                "required": False,
                "description": "Columns to aggregate.",
            },
            "agg_func": {
                "type": "string",
                "required": False,
                "default": "mean",
                "choices": self.AGG_FUNCTIONS,
                "description": "Aggregation function.",
            },
            "merge_back": {
                "type": "bool",
                "required": False,
                "default": True,
                "description": "Merge aggregated values back to original df.",
            },
            "index": {"type": "string", "required": False, "description": "Pivot index column."},
            "pivot_column": {"type": "string", "required": False, "description": "Pivot column."},
            "value_column": {"type": "string", "required": False, "description": "Pivot value column."},
            "id_vars": {"type": "list", "required": False, "description": "Melt ID vars."},
            "value_vars": {"type": "list", "required": False, "description": "Melt value vars."},
            "var_name": {"type": "string", "required": False, "default": "variable"},
            "value_name": {"type": "string", "required": False, "default": "value"},
            "columns": {"type": "list", "required": False, "description": "Columns for cumulative ops."},
            "cum_func": {
                "type": "string",
                "required": False,
                "default": "cumsum",
                "choices": self.CUMULATIVE_FUNCTIONS,
            },
        }

    def get_description(self, parameters: Dict[str, Any]) -> str:
        operation = parameters.get("operation", "unknown")
        if operation == "group_aggregate":
            return f"Group by {parameters.get('group_by')} and compute {parameters.get('agg_func', 'mean')} of {parameters.get('agg_columns')}"
        elif operation == "pivot":
            return f"Pivot: index={parameters.get('index')}, columns={parameters.get('pivot_column')}"
        elif operation == "melt":
            return f"Melt/Unpivot: id_vars={parameters.get('id_vars')}"
        elif operation == "cumulative":
            return f"Cumulative {parameters.get('cum_func', 'cumsum')} on {parameters.get('columns')}"
        return f"Aggregation: {operation}"

    def get_code_snippet(self, parameters: Dict[str, Any]) -> str:
        operation = parameters.get("operation")
        if operation == "group_aggregate":
            return f"agg_df = df.groupby({parameters.get('group_by')})[{parameters.get('agg_columns')}].agg('{parameters.get('agg_func', 'mean')}')"
        elif operation == "pivot":
            return f"df = df.pivot_table(index='{parameters.get('index')}', columns='{parameters.get('pivot_column')}', values='{parameters.get('value_column')}')"
        return f"# Aggregation: {operation}"
