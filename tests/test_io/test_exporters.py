"""
Tests for DataExporter.
"""

import pytest
import pandas as pd
import json

from data_io.exporters import DataExporter


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
        "salary": [50000.0, 60000.0, 70000.0],
    })


class TestExporters:
    def test_csv_export(self, sample_df):
        result = DataExporter.to_csv(sample_df)
        assert isinstance(result, bytes)
        text = result.decode("utf-8")
        assert "Alice" in text
        assert "name,age,salary" in text

    def test_csv_with_delimiter(self, sample_df):
        result = DataExporter.to_csv(sample_df, delimiter=";")
        text = result.decode("utf-8")
        assert ";" in text

    def test_json_export(self, sample_df):
        result = DataExporter.to_json(sample_df)
        assert isinstance(result, bytes)
        data = json.loads(result.decode("utf-8"))
        assert len(data) == 3
        assert data[0]["name"] == "Alice"

    def test_excel_export(self, sample_df):
        result = DataExporter.to_excel(sample_df)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_parquet_export(self, sample_df):
        result = DataExporter.to_parquet(sample_df)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_generic_export(self, sample_df):
        for fmt in ["csv", "json", "parquet"]:
            result = DataExporter.export(sample_df, format=fmt)
            assert isinstance(result, bytes)
            assert len(result) > 0

    def test_unsupported_format(self, sample_df):
        with pytest.raises(ValueError):
            DataExporter.export(sample_df, format="unsupported")

    def test_mime_types(self):
        assert "csv" in DataExporter.get_mime_type("csv")
        assert "json" in DataExporter.get_mime_type("json")
