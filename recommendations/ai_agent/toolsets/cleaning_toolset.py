"""Cleaning toolset prompt — injected into the agent's system prompt."""

CLEANING_TOOLSET_PROMPT = """## Available Cleaning Tools

You may ONLY recommend actions with these action_type values and their valid parameters:

### 1. handle_missing
Handle missing values in columns.
Parameters:
  - columns: list of column names
  - strategy: one of "mean", "median", "mode", "constant", "forward_fill", "backward_fill", "drop_rows", "drop_columns", "group_based"
  - fill_value: (only for "constant" strategy) the value to fill
  - group_by: (only for "group_based") column to group by
  - group_strategy: (only for "group_based") one of "mean", "median", "mode"

### 2. handle_duplicates
Remove duplicate rows.
Parameters:
  - subset: list of columns to check (null = all columns)
  - keep: one of "first", "last", "none"

### 3. handle_outliers
Handle outliers in numeric columns.
Parameters:
  - columns: list of numeric column names
  - method: one of "iqr", "zscore", "percentile"
  - threshold: numeric (1.5 for IQR, 3.0 for zscore)
  - behavior: one of "remove", "clip", "flag"

### 4. text_cleaning
Clean text/string columns.
Parameters:
  - columns: list of text column names
  - operations: list of operations from: "trim_whitespace", "lowercase", "uppercase", "titlecase", "remove_special_chars", "remove_punctuation", "collapse_whitespace", "strip_html"

### 5. inconsistency
Fix data inconsistencies.
Parameters:
  - columns: list of column names
  - operation: one of "value_mapping", "merge_rare_categories", "standardize_values"
  - mapping: (for "value_mapping") dict of old_value → new_value
  - threshold: (for "merge_rare_categories") frequency threshold (e.g., 0.01)
  - replacement: (for "merge_rare_categories") replacement value (e.g., "Other")"""