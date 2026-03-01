"""
Shared test fixtures for DataPrep Kit tests.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from core.dataset import Dataset
from core.pipeline import Pipeline


@pytest.fixture
def sample_dirty_df():
    """A sample dirty dataframe for testing."""
    return pd.DataFrame({
        "name": ["Alice", " Bob ", "CHARLIE", "alice", None, "Frank", "  Grace  "],
        "age": [25, 30, None, 25, 45, 200, 35],
        "salary": [50000, 60000, 70000, 50000, None, 80000, 55000],
        "department": ["Engineering", "marketing", "Engineering", "engineering", "Sales", "HR", "Sales"],
        "join_date": ["2022-01-15", "2022-02-20", "2021/06/10", "15-03-2022", "2020-11-01", "2023-01-30", "invalid"],
        "rating": [4.5, 3.8, 4.2, 4.5, 3.0, None, 3.9],
        "is_active": ["yes", "true", "1", "Yes", "no", "false", "0"],
    })


@pytest.fixture
def sample_numeric_df():
    """A sample numeric dataframe for testing."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "feature_a": np.random.normal(50, 10, n),
        "feature_b": np.random.normal(100, 25, n),
        "feature_c": np.random.normal(50, 10, n) * 0.95 + np.random.normal(0, 1, n),  # correlated with a
        "target": np.random.binomial(1, 0.5, n),
    })


@pytest.fixture
def sample_dataset(sample_dirty_df):
    """A Dataset object wrapping the dirty df."""
    return Dataset(
        df=sample_dirty_df,
        source_name="test_data.csv",
        import_config={"format": "csv", "delimiter": ","},
    )


@pytest.fixture
def empty_pipeline():
    """An empty pipeline."""
    return Pipeline()


@pytest.fixture
def sample_pipeline():
    """A pipeline with a few pre-defined steps."""
    pipeline = Pipeline()
    pipeline.add_action(
        action_type="handle_missing",
        description="Impute age with median",
        parameters={"columns": ["age"], "strategy": "median"},
        author="user",
    )
    pipeline.add_action(
        action_type="handle_duplicates",
        description="Remove duplicates keeping first",
        parameters={"subset": None, "keep": "first"},
        author="ai_static",
    )
    return pipeline