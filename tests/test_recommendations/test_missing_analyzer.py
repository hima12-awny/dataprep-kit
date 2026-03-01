"""
Tests for the MissingAnalyzer static recommendation engine.
"""

import pytest
import pandas as pd
import numpy as np

from recommendations.static.missing_analyzer import MissingAnalyzer


@pytest.fixture
def analyzer():
    return MissingAnalyzer()


class TestMissingAnalyzer:
    def test_no_missing(self, analyzer):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        recs = analyzer.analyze(df)
        assert len(recs) == 0

    def test_high_missing_suggests_drop(self, analyzer):
        # >50% missing → should suggest drop
        df = pd.DataFrame({"a": [1, None, None, None, None, None, None, None, None, None]})
        recs = analyzer.analyze(df)
        assert len(recs) > 0
        assert recs[0]["parameters"]["strategy"] == "drop_columns"
        assert recs[0]["priority"] == "high"

    def test_medium_missing_numeric_suggests_impute(self, analyzer):
        # ~20% missing numeric → should suggest median or mean
        values = list(range(80)) + [None] * 20
        df = pd.DataFrame({"age": values})
        recs = analyzer.analyze(df)
        assert len(recs) > 0
        assert recs[0]["parameters"]["strategy"] in ("mean", "median")
        assert recs[0]["priority"] == "medium"

    def test_medium_missing_categorical_suggests_mode(self, analyzer):
        # ~20% missing categorical → should suggest mode
        values = ["A"] * 40 + ["B"] * 40 + [None] * 20
        df = pd.DataFrame({"category": values})
        recs = analyzer.analyze(df)
        assert len(recs) > 0
        assert recs[0]["parameters"]["strategy"] == "mode"

    def test_low_missing(self, analyzer):
        # ~2% missing → low priority
        values = list(range(98)) + [None] * 2
        df = pd.DataFrame({"value": values})
        recs = analyzer.analyze(df)
        assert len(recs) > 0
        assert recs[0]["priority"] == "low"

    def test_recommendation_structure(self, analyzer):
        df = pd.DataFrame({"a": [1, None, 3]})
        recs = analyzer.analyze(df)
        assert len(recs) > 0
        rec = recs[0]
        assert "action_id" in rec
        assert "action_type" in rec
        assert "description" in rec
        assert "author" in rec
        assert rec["author"] == "ai_static"
        assert "parameters" in rec
        assert "priority" in rec
        assert "reason" in rec

    def test_skewed_numeric_suggests_median(self, analyzer):
        # Highly skewed → should prefer median
        np.random.seed(42)
        values = list(np.random.exponential(scale=100, size=80)) + [None] * 20
        df = pd.DataFrame({"income": values})
        recs = analyzer.analyze(df)
        assert len(recs) > 0
        # Exponential is skewed, so should suggest median
        assert recs[0]["parameters"]["strategy"] == "median"

    def test_empty_dataframe(self, analyzer):
        df = pd.DataFrame()
        recs = analyzer.analyze(df)
        assert len(recs) == 0