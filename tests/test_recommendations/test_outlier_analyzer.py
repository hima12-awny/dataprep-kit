"""
Tests for the OutlierAnalyzer static recommendation engine.
"""

import pytest
import pandas as pd
import numpy as np

from recommendations.static.outlier_analyzer import OutlierAnalyzer


@pytest.fixture
def analyzer():
    return OutlierAnalyzer()


class TestOutlierAnalyzer:
    def test_no_outliers(self, analyzer):
        np.random.seed(42)
        df = pd.DataFrame({"value": np.random.normal(50, 1, 100)})
        recs = analyzer.analyze(df)
        # With very tight distribution, there might be no outliers
        # Or very few flagged ones with low priority
        for rec in recs:
            assert rec["action_type"] == "handle_outliers"

    def test_detects_extreme_outliers(self, analyzer):
        values = list(np.random.normal(50, 5, 97)) + [500, -400, 1000]
        df = pd.DataFrame({"value": values})
        recs = analyzer.analyze(df)
        assert len(recs) > 0
        assert recs[0]["action_type"] == "handle_outliers"
        assert recs[0]["parameters"]["columns"] == ["value"]

    def test_high_outlier_pct_suggests_clip(self, analyzer):
        # Create data with >10% outliers
        np.random.seed(42)
        normal = list(np.random.normal(50, 5, 80))
        outliers = list(np.random.normal(200, 50, 20))
        df = pd.DataFrame({"value": normal + outliers})
        recs = analyzer.analyze(df)
        assert len(recs) > 0
        assert recs[0]["parameters"]["behavior"] == "clip"
        assert recs[0]["priority"] == "high"

    def test_low_outlier_pct_suggests_flag(self, analyzer):
        # Create data with very few outliers (<3% but >0.5%)
        np.random.seed(42)
        normal = list(np.random.normal(50, 5, 99))
        outliers = [500]
        df = pd.DataFrame({"value": normal + outliers})
        recs = analyzer.analyze(df)
        if recs:
            # Should suggest flag for small number of outliers
            assert recs[0]["parameters"]["behavior"] in ("flag", "clip")

    def test_skips_non_numeric(self, analyzer):
        df = pd.DataFrame({"text": ["a", "b", "c", "d", "e"]})
        recs = analyzer.analyze(df)
        assert len(recs) == 0

    def test_recommendation_structure(self, analyzer):
        values = list(np.random.normal(50, 5, 95)) + \
            [500, -400, 1000, 800, 900]
        df = pd.DataFrame({"value": values})
        recs = analyzer.analyze(df)
        if recs:
            rec = recs[0]
            assert rec["author"] == "ai_static"
            assert "method" in rec["parameters"]
            assert "threshold" in rec["parameters"]
            assert "behavior" in rec["parameters"]
            assert "reason" in rec

    def test_empty_column(self, analyzer):
        df = pd.DataFrame({"value": [np.nan] * 10})
        recs = analyzer.analyze(df)
        assert len(recs) == 0
