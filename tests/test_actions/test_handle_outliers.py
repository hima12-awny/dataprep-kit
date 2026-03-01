"""
Tests for HandleOutliersAction.
"""

import pytest
import pandas as pd
import numpy as np

from actions.cleaning.handle_outliers import HandleOutliersAction


@pytest.fixture
def action():
    return HandleOutliersAction()


@pytest.fixture
def df_with_outliers():
    np.random.seed(42)
    normal_data = np.random.normal(50, 10, 100)
    # Add outliers
    normal_data[0] = 200
    normal_data[1] = -100
    normal_data[2] = 500
    return pd.DataFrame({"value": normal_data, "category": ["A"] * 50 + ["B"] * 50})


class TestOutlierValidation:
    def test_valid_iqr(self, action, df_with_outliers):
        errors = action.validate(df_with_outliers, {
            "columns": ["value"], "method": "iqr", "behavior": "clip"
        })
        assert len(errors) == 0

    def test_non_numeric_column(self, action, df_with_outliers):
        errors = action.validate(df_with_outliers, {
            "columns": ["category"], "method": "iqr", "behavior": "clip"
        })
        assert len(errors) > 0

    def test_invalid_method(self, action, df_with_outliers):
        errors = action.validate(df_with_outliers, {
            "columns": ["value"], "method": "invalid", "behavior": "clip"
        })
        assert len(errors) > 0


class TestOutlierExecution:
    def test_clip(self, action, df_with_outliers):
        result = action.execute(df_with_outliers, {
            "columns": ["value"], "method": "iqr", "threshold": 1.5, "behavior": "clip"
        })
        assert result["value"].max() < df_with_outliers["value"].max()

    def test_remove(self, action, df_with_outliers):
        result = action.execute(df_with_outliers, {
            "columns": ["value"], "method": "iqr", "threshold": 1.5, "behavior": "remove"
        })
        assert len(result) < len(df_with_outliers)

    def test_flag(self, action, df_with_outliers):
        result = action.execute(df_with_outliers, {
            "columns": ["value"], "method": "iqr", "threshold": 1.5, "behavior": "flag"
        })
        assert "value_outlier_flag" in result.columns
        assert result["value_outlier_flag"].sum() > 0

    def test_does_not_mutate(self, action, df_with_outliers):
        original_max = df_with_outliers["value"].max()
        action.execute(df_with_outliers, {
            "columns": ["value"], "method": "iqr", "behavior": "clip"
        })
        assert df_with_outliers["value"].max() == original_max