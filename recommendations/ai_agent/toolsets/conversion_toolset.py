"""Conversion toolset prompt — injected into the agent's system prompt."""

CONVERSION_TOOLSET_PROMPT = """## Available Conversion Tools

You may ONLY recommend actions with these action_type values and their valid parameters:

### 1. type_casting
Convert column data types.
Parameters:
  - conversions: dict of {column_name: target_type}
  - target_type must be one of: "int64", "float64", "string", "bool", "datetime64[ns]", "category"
  - errors: one of "coerce", "raise", "ignore" (default: "coerce")
  - date_format: (optional) explicit format string like "%Y-%m-%d"

### 2. datetime_ops
Datetime operations.
Parameters:
  - operation: one of "extract_components", "date_diff", "to_unix_timestamp", "from_unix_timestamp"
  - column: source datetime column name
  - components: (for "extract_components") list from: "year", "month", "day", "weekday", "day_name", "quarter", "week", "is_weekend", "is_month_start", "is_month_end", "hour", "minute"
  - column2: (for "date_diff") second column
  - unit: (for "date_diff") one of "days", "hours", "seconds"

### 3. numeric_transform
Numeric transformations.
Parameters:
  - columns: list of numeric column names
  - operation: one of "log", "log1p", "sqrt", "square", "abs", "round", "clip", "normalize", "standardize", "robust_scale", "binning"
  - overwrite: boolean (default: true)
  - n_bins: (for "binning") integer

### 4. encoding
Categorical encoding.
Parameters:
  - columns: list of categorical column names
  - method: one of "label", "onehot", "frequency", "ordinal", "binary", "target"
  - drop_first: (for "onehot") boolean
  - drop_original: boolean (default: true)
  - target_column: (for "target" encoding) name of the target column"""
