"""
Tests for HandleMissingAction.
"""

import pytest
import pandas as pd
import numpy as np

from actions.cleaning.handle_missing import HandleMissingAction


@pytest.fixture
def action():
    return HandleMissingAction()


@pytest.fixture
def df_with_missing():
    return pd.DataFrame({
        "age": [25, None, 35, None, 45],
        "name": ["Alice", None, "Charlie", "Diana", None],
        "salary": [50000, 60000, None, 70000, 80000],
    })


class TestHandleMissingValidation:
    def test_valid_mean(self, action, df_with_missing):
        errors = action.validate(df_with_missing, {"columns": ["age"], "strategy": "mean"})
        assert len(errors) == 0

    def test_invalid_strategy(self, action, df_with_missing):
        errors = action.validate(df_with_missing, {"columns": ["age"], "strategy": "invalid"})
        assert len(errors) > 0

    def test_missing_column(self, action, df_with_missing):
        errors = action.validate(df_with_missing, {"columns": ["nonexistent"], "strategy": "mean"})
        assert len(errors) > 0

    def test_mean_on_non_numeric(self, action, df_with_missing):
        errors = action.validate(df_with_missing, {"columns": ["name"], "strategy": "mean"})
        assert len(errors) > 0

    def test_constant_without_value(self, action, df_with_missing):
        errors = action.validate(df_with_missing, {"columns": ["age"], "strategy": "constant"})
        assert len(errors) > 0


class TestHandleMissingExecution:
    def test_mean_imputation(self, action, df_with_missing):
        result = action.execute(df_with_missing, {"columns": ["age"], "strategy": "mean"})
        assert result["age"].isna().sum() == 0
        assert df_with_missing["age"].isna().sum() > 0  # Original unchanged

    def test_median_imputation(self, action, df_with_missing):
        result = action.execute(df_with_missing, {"columns": ["salary"], "strategy": "median"})
        assert result["salary"].isna().sum() == 0

    def test_mode_imputation(self, action, df_with_missing):
        result = action.execute(df_with_missing, {"columns": ["name"], "strategy": "mode"})
        assert result["name"].isna().sum() == 0

    def test_constant_imputation(self, action, df_with_missing):
        result = action.execute(df_with_missing, {"columns": ["name"], "strategy": "constant", "fill_value": "Unknown"})
        assert result["name"].isna().sum() == 0
        assert "Unknown" in result["name"].values

    def test_drop_rows(self, action, df_with_missing):
        result = action.execute(df_with_missing, {"columns": ["age"], "strategy": "drop_rows"})
        assert len(result) < len(df_with_missing)
        assert result["age"].isna().sum() == 0

    def test_forward_fill(self, action, df_with_missing):
        result = action.execute(df_with_missing, {"columns": ["age"], "strategy": "forward_fill"})
        # First value is 25, so ffill should fill forward
        assert result["age"].iloc[1] == 25

    def test_does_not_mutate_original(self, action, df_with_missing):
        original_nulls = df_with_missing["age"].isna().sum()
        action.execute(df_with_missing, {"columns": ["age"], "strategy": "mean"})
        assert df_with_missing["age"].isna().sum() == original_nulls


class TestHandleMissingSerialization:
    def test_serialize(self, action):
        result = action.serialize(
            parameters={"columns": ["age"], "strategy": "median"},
            author="user",
        )
        assert result["action_type"] == "handle_missing"
        assert result["author"] == "user"
        assert "action_id" in result
        assert "timestamp" in result

    def test_get_description(self, action):
        desc = action.get_description({"columns": ["age", "salary"], "strategy": "mean"})
        assert "age" in desc
        assert "mean" in desc

    def test_get_code_snippet(self, action):
        code = action.get_code_snippet({"columns": ["age"], "strategy": "mean"})
        assert "fillna" in code or "mean" in code