"""
Tests for ColumnOpsAction.
"""

import pytest
import pandas as pd

from actions.feature_engineering.column_ops import ColumnOpsAction


@pytest.fixture
def action():
    return ColumnOpsAction()


@pytest.fixture
def df_basic():
    return pd.DataFrame({
        "col_a": [10, 20, 30, 40, 50],
        "col_b": [2, 4, 6, 8, 10],
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "city": ["NY", "LA", "SF", "NY", "LA"],
    })


class TestColumnOpsValidation:
    def test_valid_create_expression(self, action, df_basic):
        errors = action.validate(df_basic, {
            "operation": "create_expression",
            "new_column": "result",
            "expression": "col_a + col_b",
        })
        assert len(errors) == 0

    def test_create_expression_missing_name(self, action, df_basic):
        errors = action.validate(df_basic, {
            "operation": "create_expression",
            "expression": "col_a + col_b",
        })
        assert len(errors) > 0

    def test_create_expression_missing_expr(self, action, df_basic):
        errors = action.validate(df_basic, {
            "operation": "create_expression",
            "new_column": "result",
        })
        assert len(errors) > 0

    def test_combine_needs_two_columns(self, action, df_basic):
        errors = action.validate(df_basic, {
            "operation": "combine_columns",
            "columns": ["name"],
        })
        assert len(errors) > 0

    def test_split_missing_delimiter(self, action, df_basic):
        errors = action.validate(df_basic, {
            "operation": "split_column",
            "column": "name",
        })
        assert len(errors) > 0

    def test_drop_missing_columns(self, action, df_basic):
        errors = action.validate(df_basic, {
            "operation": "drop_columns",
            "columns": ["nonexistent"],
        })
        assert len(errors) > 0

    def test_invalid_operation(self, action, df_basic):
        errors = action.validate(df_basic, {
            "operation": "invalid_op",
        })
        assert len(errors) > 0


class TestColumnOpsExecution:
    def test_create_expression(self, action, df_basic):
        result = action.execute(df_basic, {
            "operation": "create_expression",
            "new_column": "ratio",
            "expression": "col_a / col_b",
        })
        assert "ratio" in result.columns
        assert result["ratio"].iloc[0] == 5.0  # 10 / 2

    def test_create_expression_with_numpy(self, action, df_basic):
        result = action.execute(df_basic, {
            "operation": "create_expression",
            "new_column": "log_a",
            "expression": "np.log(col_a)",
        })
        assert "log_a" in result.columns

    def test_combine_columns(self, action, df_basic):
        result = action.execute(df_basic, {
            "operation": "combine_columns",
            "columns": ["name", "city"],
            "separator": " - ",
            "new_column": "name_city",
        })
        assert "name_city" in result.columns
        assert result["name_city"].iloc[0] == "Alice - NY"

    def test_split_column(self, action):
        df = pd.DataFrame(
            {"full_name": ["Alice Smith", "Bob Jones", "Charlie Brown"]})
        result = action.execute(df, {
            "operation": "split_column",
            "column": "full_name",
            "delimiter": " ",
            "prefix": "name",
        })
        assert "name_0" in result.columns
        assert "name_1" in result.columns
        assert result["name_0"].iloc[0] == "Alice"

    def test_rename_columns(self, action, df_basic):
        result = action.execute(df_basic, {
            "operation": "rename_columns",
            "rename_map": {"col_a": "feature_a", "col_b": "feature_b"},
        })
        assert "feature_a" in result.columns
        assert "col_a" not in result.columns

    def test_drop_columns(self, action, df_basic):
        result = action.execute(df_basic, {
            "operation": "drop_columns",
            "columns": ["col_a", "col_b"],
        })
        assert "col_a" not in result.columns
        assert "col_b" not in result.columns
        assert "name" in result.columns

    def test_reorder_columns(self, action, df_basic):
        result = action.execute(df_basic, {
            "operation": "reorder_columns",
            "new_order": ["city", "name"],
        })
        assert list(result.columns[:2]) == ["city", "name"]

    def test_does_not_mutate_original(self, action, df_basic):
        original_cols = list(df_basic.columns)
        action.execute(df_basic, {
            "operation": "drop_columns",
            "columns": ["col_a"],
        })
        assert list(df_basic.columns) == original_cols
