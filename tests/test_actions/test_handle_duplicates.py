"""
Tests for HandleDuplicatesAction.
"""

import pytest
import pandas as pd

from actions.cleaning.handle_duplicates import HandleDuplicatesAction


@pytest.fixture
def action():
    return HandleDuplicatesAction()


@pytest.fixture
def df_with_dups():
    return pd.DataFrame({
        "a": [1, 2, 1, 3, 2],
        "b": ["x", "y", "x", "z", "y"],
    })


class TestDuplicateExecution:
    def test_keep_first(self, action, df_with_dups):
        result = action.execute(df_with_dups, {"keep": "first"})
        assert len(result) == 3

    def test_keep_last(self, action, df_with_dups):
        result = action.execute(df_with_dups, {"keep": "last"})
        assert len(result) == 3

    def test_keep_none(self, action, df_with_dups):
        result = action.execute(df_with_dups, {"keep": "none"})
        assert len(result) == 1  # Only row with (3, "z") is not duplicated

    def test_subset(self, action, df_with_dups):
        result = action.execute(df_with_dups, {"subset": ["a"], "keep": "first"})
        assert len(result) == 3

    def test_preview(self, action, df_with_dups):
        preview = action.preview(df_with_dups, {"keep": "first"})
        assert preview["duplicate_count"] == 4  # 2 pairs = 4 rows involved
        assert preview["rows_after"] < preview["rows_before"]