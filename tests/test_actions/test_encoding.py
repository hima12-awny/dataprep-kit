"""
Tests for EncodingAction.
"""

import pytest
import pandas as pd

from actions.conversion.encoding import EncodingAction


@pytest.fixture
def action():
    return EncodingAction()


@pytest.fixture
def df_categorical():
    return pd.DataFrame({
        "color": ["red", "blue", "green", "red", "blue", "green"],
        "size": ["S", "M", "L", "M", "S", "L"],
        "value": [10, 20, 30, 40, 50, 60],
    })


class TestEncoding:
    def test_label_encoding(self, action, df_categorical):
        result = action.execute(df_categorical, {"columns": ["color"], "method": "label"})
        assert pd.api.types.is_numeric_dtype(result["color"])

    def test_onehot_encoding(self, action, df_categorical):
        result = action.execute(df_categorical, {
            "columns": ["color"], "method": "onehot", "drop_original": True
        })
        assert "color" not in result.columns
        assert any("color_" in col for col in result.columns)

    def test_onehot_drop_first(self, action, df_categorical):
        result = action.execute(df_categorical, {
            "columns": ["color"], "method": "onehot",
            "drop_first": True, "drop_original": True
        })
        color_cols = [c for c in result.columns if c.startswith("color_")]
        # drop_first should produce n-1 columns
        assert len(color_cols) == 2  # 3 categories - 1

    def test_frequency_encoding(self, action, df_categorical):
        result = action.execute(df_categorical, {
            "columns": ["color"], "method": "frequency"
        })
        assert "color_freq_encoded" in result.columns

    def test_target_encoding(self, action, df_categorical):
        result = action.execute(df_categorical, {
            "columns": ["color"], "method": "target", "target_column": "value"
        })
        assert "color_target_encoded" in result.columns

    def test_validation_missing_method(self, action, df_categorical):
        errors = action.validate(df_categorical, {"columns": ["color"]})
        assert len(errors) > 0

    def test_validation_invalid_method(self, action, df_categorical):
        errors = action.validate(df_categorical, {"columns": ["color"], "method": "invalid"})
        assert len(errors) > 0

    def test_validation_target_without_target_col(self, action, df_categorical):
        errors = action.validate(df_categorical, {"columns": ["color"], "method": "target"})
        assert len(errors) > 0

    def test_validation_ordinal_without_order(self, action, df_categorical):
        errors = action.validate(df_categorical, {"columns": ["color"], "method": "ordinal"})
        assert len(errors) > 0

    def test_ordinal_encoding(self, action, df_categorical):
        result = action.execute(df_categorical, {
            "columns": ["size"],
            "method": "ordinal",
            "order": {"size": ["S", "M", "L"]},
        })
        assert result["size"].iloc[0] == 0  # S → 0
        assert result["size"].iloc[1] == 1  # M → 1
        assert result["size"].iloc[2] == 2  # L → 2

    def test_binary_encoding(self, action, df_categorical):
        result = action.execute(df_categorical, {
            "columns": ["color"],
            "method": "binary",
            "drop_original": True,
        })
        assert "color" not in result.columns
        bit_cols = [c for c in result.columns if c.startswith("color_bit_")]
        assert len(bit_cols) > 0

    def test_does_not_mutate_original(self, action, df_categorical):
        original_cols = list(df_categorical.columns)
        action.execute(df_categorical, {
            "columns": ["color"], "method": "onehot", "drop_original": True
        })
        assert list(df_categorical.columns) == original_cols