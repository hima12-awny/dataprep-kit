"""
Tests for the Dataset class: snapshots, undo/redo.
"""

import pytest
import pandas as pd

from core.dataset import Dataset


class TestDataset:
    def test_creation(self, sample_dirty_df):
        ds = Dataset(df=sample_dirty_df, source_name="test.csv")
        assert ds.shape == sample_dirty_df.shape
        assert ds.source_name == "test.csv"

    def test_properties(self, sample_dataset):
        assert len(sample_dataset.columns) > 0
        assert len(sample_dataset.numeric_columns) > 0
        assert len(sample_dataset.categorical_columns) > 0

    def test_update_and_undo(self, sample_dataset):
        original_shape = sample_dataset.shape
        new_df = sample_dataset.df.dropna()
        sample_dataset.update(new_df, action_id="test1", description="Drop nulls")

        assert sample_dataset.shape != original_shape
        assert sample_dataset.can_undo

        sample_dataset.undo()
        assert sample_dataset.shape == original_shape

    def test_redo(self, sample_dataset):
        new_df = sample_dataset.df.dropna()
        dropped_shape = new_df.shape
        sample_dataset.update(new_df, action_id="test1", description="Drop nulls")
        sample_dataset.undo()
        sample_dataset.redo()
        assert sample_dataset.shape == dropped_shape

    def test_cannot_undo_past_initial(self, sample_dataset):
        assert not sample_dataset.can_undo or sample_dataset.undo_steps_count == 0
        result = sample_dataset.undo()
        assert result is False

    def test_overview(self, sample_dataset):
        overview = sample_dataset.overview()
        assert "row_count" in overview
        assert "column_count" in overview
        assert "missing_percentage" in overview
        assert "duplicate_rows" in overview

    def test_column_stats(self, sample_dataset):
        stats = sample_dataset.column_stats("age")
        assert "dtype" in stats
        assert "null_count" in stats

    def test_metadata(self, sample_dataset):
        meta = sample_dataset.get_metadata()
        assert meta["source_name"] == "test_data.csv"
        assert "original_shape" in meta
        assert "current_shape" in meta