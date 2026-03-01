"""
Tests for TypeDetector.
"""

import pytest
import pandas as pd

from utils.type_detector import TypeDetector


class TestTypeDetector:
    def test_detect_date_column(self):
        series = pd.Series(["2022-01-01", "2022-02-15", "2022-03-30", "2022-04-10", "2022-05-20"])
        result = TypeDetector.analyze_column(series)
        assert result["suggested_type"] == "datetime64[ns]"
        assert result["confidence"] >= 0.7

    def test_detect_integer_as_string(self):
        series = pd.Series(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
        result = TypeDetector.analyze_column(series)
        assert result["suggested_type"] == "int64"
        assert result["confidence"] >= 0.9

    def test_detect_float_as_string(self):
        series = pd.Series(["1.1", "2.2", "3.3", "4.4", "5.5", "6.6", "7.7", "8.8", "9.9"])
        result = TypeDetector.analyze_column(series)
        assert result["suggested_type"] in ("float64", "int64")  # Could detect as either

    def test_detect_boolean_as_string(self):
        series = pd.Series(["true", "false", "true", "false", "true", "false"])
        result = TypeDetector.analyze_column(series)
        assert result["suggested_type"] == "bool"
        assert result["confidence"] >= 0.9

    def test_detect_categorical(self):
        series = pd.Series(["A"] * 50 + ["B"] * 30 + ["C"] * 20)
        result = TypeDetector.analyze_column(series)
        assert result["suggested_type"] == "category"

    def test_no_suggestion_for_genuine_text(self):
        series = pd.Series([
            "The quick brown fox",
            "jumps over the lazy dog",
            "Lorem ipsum dolor sit amet",
            "Another unique sentence here",
            "Yet another completely different text",
        ] * 20)
        result = TypeDetector.analyze_column(series)
        # Should not suggest numeric or boolean
        if result["suggested_type"]:
            assert result["suggested_type"] in ("category",)  # Could be category due to repeats

    def test_empty_series(self):
        series = pd.Series([None, None, None], dtype="object")
        result = TypeDetector.analyze_column(series)
        assert result["confidence"] == 0.0

    def test_numeric_as_categorical(self):
        # Numeric column with very few unique values
        series = pd.Series([1, 2, 3, 1, 2, 3, 1, 2, 3] * 20)
        result = TypeDetector.analyze_column(series)
        assert result["suggested_type"] == "category"

    def test_dataframe_analysis(self):
        df = pd.DataFrame({
            "dates": ["2022-01-01", "2022-02-01", "2022-03-01"],
            "nums": ["100", "200", "300"],
            "text": ["hello world", "foo bar", "baz qux"],
        })
        results = TypeDetector.analyze_dataframe(df)
        assert "dates" in results
        assert "nums" in results
        assert "text" in results

    def test_get_suggested_conversions(self):
        df = pd.DataFrame({
            "date_col": ["2022-01-01", "2022-02-01", "2022-03-01", "2022-04-01", "2022-05-01"],
            "int_col": ["1", "2", "3", "4", "5"],
            "actual_int": [1, 2, 3, 4, 5],
        })
        suggestions = TypeDetector.get_suggested_conversions(df)
        assert len(suggestions) >= 2  # date_col and int_col
        suggested_cols = [s["column"] for s in suggestions]
        assert "date_col" in suggested_cols
        assert "int_col" in suggested_cols