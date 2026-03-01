"""
Tests for TypeCastingAction.
"""

import pytest
import pandas as pd

from actions.conversion.type_casting import TypeCastingAction


@pytest.fixture
def action():
    return TypeCastingAction()


@pytest.fixture
def df_mixed():
    return pd.DataFrame({
        "num_str": ["1", "2", "3", "4", "5"],
        "float_str": ["1.1", "2.2", "3.3", "bad", "5.5"],
        "date_str": ["2022-01-01", "2022-02-01", "2022-03-01", "2022-04-01", "2022-05-01"],
        "bool_str": ["true", "false", "yes", "no", "1"],
        "already_int": [1, 2, 3, 4, 5],
    })


class TestTypeCasting:
    def test_str_to_int(self, action, df_mixed):
        result = action.execute(df_mixed, {"conversions": {"num_str": "int64"}, "errors": "coerce"})
        assert pd.api.types.is_integer_dtype(result["num_str"])

    def test_str_to_float_with_errors(self, action, df_mixed):
        result = action.execute(df_mixed, {"conversions": {"float_str": "float64"}, "errors": "coerce"})
        assert pd.api.types.is_float_dtype(result["float_str"])
        assert result["float_str"].isna().sum() == 1  # "bad" becomes NaN

    def test_str_to_datetime(self, action, df_mixed):
        result = action.execute(df_mixed, {"conversions": {"date_str": "datetime64[ns]"}, "errors": "coerce"})
        assert pd.api.types.is_datetime64_any_dtype(result["date_str"])
        
    def test_str_to_bool(self, action, df_mixed):
        result = action.execute(df_mixed, {"conversions": {"bool_str": "bool"}, "errors": "coerce"})
        assert result["bool_str"].iloc[0] == True
        assert result["bool_str"].iloc[1] == False

    def test_to_category(self, action, df_mixed):
        result = action.execute(df_mixed, {"conversions": {"num_str": "category"}, "errors": "coerce"})
        assert result["num_str"].dtype.name == "category"

    def test_validation_missing_column(self, action, df_mixed):
        errors = action.validate(df_mixed, {"conversions": {"nonexistent": "int64"}})
        assert len(errors) > 0

    def test_validation_invalid_type(self, action, df_mixed):
        errors = action.validate(df_mixed, {"conversions": {"num_str": "invalid_type"}})
        assert len(errors) > 0