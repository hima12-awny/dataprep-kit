"""
Tests for the TypeAnalyzer static recommendation engine.
"""

import pytest
import pandas as pd

from recommendations.static.type_analyzer import TypeAnalyzer


@pytest.fixture
def analyzer():
    return TypeAnalyzer()


class TestTypeAnalyzer:
    def test_detects_date_string(self, analyzer):
        df = pd.DataFrame({
            "date_col": ["2022-01-15", "2022-02-20", "2022-03-10", "2022-04-05", "2022-05-18"]
        })
        recs = analyzer.analyze(df)
        assert len(recs) > 0
        assert recs[0]["parameters"]["conversions"]["date_col"] == "datetime64[ns]"

    def test_detects_numeric_string(self, analyzer):
        df = pd.DataFrame({
            "num_col": ["100", "200", "300", "400", "500"]
        })
        recs = analyzer.analyze(df)
        assert len(recs) > 0
        target = recs[0]["parameters"]["conversions"]["num_col"]
        assert target in ("int64", "float64")

    def test_detects_boolean_string(self, analyzer):
        df = pd.DataFrame({
            "bool_col": ["true", "false", "true", "false", "true"]
        })
        recs = analyzer.analyze(df)
        assert len(recs) > 0
        assert recs[0]["parameters"]["conversions"]["bool_col"] == "bool"

    def test_no_suggestion_for_already_correct(self, analyzer):
        df = pd.DataFrame({
            "int_col": [1, 2, 3, 4, 5],
            "text_col": ["This is a sentence", "Another one here", "Yet another text",
                         "Unique text value", "Different text here"],
        })
        recs = analyzer.analyze(df)
        # int_col is already int, text_col is genuinely text
        # There may be a category suggestion for text_col if low cardinality
        type_conversions = [r for r in recs if "int_col" in str(r.get("parameters", {}).get("conversions", {}))]
        assert len(type_conversions) == 0

    def test_recommendation_has_confidence(self, analyzer):
        df = pd.DataFrame({
            "date_col": ["2022-01-01", "2022-02-01", "2022-03-01"]
        })
        recs = analyzer.analyze(df)
        if recs:
            assert "confidence" in recs[0]["description"].lower() or "%" in recs[0]["description"]

    def test_recommendation_structure(self, analyzer):
        df = pd.DataFrame({
            "num_str": ["1", "2", "3", "4", "5"]
        })
        recs = analyzer.analyze(df)
        assert len(recs) > 0
        rec = recs[0]
        assert rec["action_type"] == "type_casting"
        assert rec["author"] == "ai_static"
        assert "conversions" in rec["parameters"]