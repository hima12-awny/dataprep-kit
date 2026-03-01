"""Feature engineering toolset prompt — injected into the agent's system prompt."""

ENGINEERING_TOOLSET_PROMPT = """## Available Feature Engineering Tools

You may ONLY recommend actions with these action_type values and their valid parameters:

### 1. column_ops
Column operations.
Parameters:
  - operation: one of "create_expression", "combine_columns", "split_column", "rename_columns", "drop_columns"
  - columns: (for combine/drop) list of column names
  - column: (for split) single column name
  - new_column: name for newly created column
  - expression: (for "create_expression") Python expression using column names and np/pd
  - separator: (for "combine_columns") string separator
  - delimiter: (for "split_column") split delimiter

### 2. aggregation
Aggregation features.
Parameters:
  - operation: one of "group_aggregate", "cumulative", "pivot", "melt"
  - group_by: list of columns to group by
  - agg_columns: list of columns to aggregate
  - agg_func: one of "sum", "mean", "median", "min", "max", "count", "std"
  - merge_back: boolean (default: true) — merge aggregated values back to original

### 3. temporal
Temporal features.
Parameters:
  - operation: one of "lag", "lead", "rolling", "cyclical_encoding"
  - columns: (for lag/lead/rolling) list of column names
  - column: (for cyclical) single column name
  - periods: (for lag/lead) integer
  - window: (for rolling) integer >= 2
  - rolling_func: one of "mean", "sum", "std", "min", "max", "median"
  - max_value: (for cyclical) max cycle value (e.g., 24 for hours, 7 for weekdays)
  - group_by: (optional) grouping column

### 4. interaction
Interaction features.
Parameters:
  - operation: one of "polynomial", "pairwise_ratio", "pairwise_difference", "pairwise_product", "cross_categorical"
  - columns: list of column names (numeric for math ops, categorical for cross)
  - degree: (for "polynomial") integer 2-4
  - separator: (for "cross_categorical") string separator"""