"""
Column selector widget with search, multi-select, and type-based filtering.
"""

import streamlit as st
import pandas as pd
from typing import List, Optional


def render_column_selector(
    df: pd.DataFrame,
    key: str = "col_selector",
    label: str = "Select columns",
    default: Optional[List[str]] = None,
    type_filter: Optional[str] = None,
    max_selections: Optional[int] = None,
    single: bool = False,
) -> List[str]:
    """
    Render an advanced column selector.

    Args:
        df: The dataframe to select columns from.
        key: Streamlit widget key.
        label: Label for the selector.
        default: Default selected columns.
        type_filter: Filter by type ('numeric', 'categorical', 'datetime', 'all').
        max_selections: Max number of columns that can be selected.
        single: If True, use selectbox instead of multiselect.

    Returns:
        List of selected column names.
    """
    # Filter columns by type
    if type_filter == "numeric":
        available = df.select_dtypes(include=["number"]).columns.tolist()
    elif type_filter == "categorical":
        available = df.select_dtypes(include=["object", "category"]).columns.tolist()
    elif type_filter == "datetime":
        available = df.select_dtypes(include=["datetime64"]).columns.tolist()
    elif type_filter == "text":
        available = df.select_dtypes(include=["object"]).columns.tolist()
    else:
        available = df.columns.tolist()

    if not available:
        st.warning(f"No {type_filter or ''} columns available.")
        return []

    # Add type annotations to column names for display
    col_info = {}
    for col in available:
        dtype_str = str(df[col].dtype)
        null_count = df[col].isna().sum()
        null_str = f", {null_count} nulls" if null_count > 0 else ""
        col_info[col] = f"{col}  ({dtype_str}{null_str})"

    # Type filter toggle
    if type_filter is None and len(df.columns) > 10:
        filter_col1, filter_col2 = st.columns([3, 1])
        with filter_col2:
            type_choice = st.selectbox(
                "Filter type",
                ["All", "Numeric", "Categorical", "Datetime"],
                key=f"{key}_type_choice",
            )
            if type_choice != "All":
                type_map = {
                    "Numeric": "number",
                    "Categorical": ["object", "category"],
                    "Datetime": "datetime64",
                }
                available = df.select_dtypes(include=type_map[type_choice]).columns.tolist()

    if single:
        selected = st.selectbox(
            label,
            options=available,
            index=available.index(default[0]) if default and default[0] in available else 0,
            key=key,
        )
        return [selected] if selected else []
    else:
        valid_default = [d for d in (default or []) if d in available]
        selected = st.multiselect(
            label,
            options=available,
            default=valid_default,
            key=key,
            max_selections=max_selections,
        )
        return selected