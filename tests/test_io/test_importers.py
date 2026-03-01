"""
Tests for DataImporter.
"""

import pytest
import pandas as pd
import json

from data_io.importers import DataImporter


class TestCSVImport:
    def test_basic_csv(self):
        csv_content = b"name,age,salary\nAlice,25,50000\nBob,30,60000\n"
        df, config = DataImporter.import_csv(csv_content)
        assert len(df) == 2
        assert list(df.columns) == ["name", "age", "salary"]
        assert config["format"] == "csv"

    def test_csv_with_delimiter(self):
        csv_content = b"name;age;salary\nAlice;25;50000\nBob;30;60000\n"
        df, config = DataImporter.import_csv(csv_content, delimiter=";")
        assert len(df) == 2

    def test_csv_with_sample(self):
        lines = "name,age\n" + "\n".join(f"Person{i},{i}" for i in range(100))
        csv_content = lines.encode("utf-8")
        df, config = DataImporter.import_csv(csv_content, sample_rows=10)
        assert len(df) == 10

    def test_csv_na_values(self):
        csv_content = b"name,age\nAlice,25\nBob,NA\nCharlie,null\n"
        df, config = DataImporter.import_csv(csv_content)
        assert df["age"].isna().sum() == 2


class TestJSONImport:
    def test_basic_json(self):
        data = [{"name": "Alice", "age": 25}, {"name": "Bob", "age": 30}]
        json_content = json.dumps(data).encode("utf-8")
        df, config = DataImporter.import_json(json_content, orient="records")
        assert len(df) == 2
        assert config["format"] == "json"


class TestAutoImport:
    def test_auto_csv(self):
        csv_content = b"a,b\n1,2\n3,4\n"
        df, config = DataImporter.auto_import(csv_content, "data.csv")
        assert len(df) == 2

    def test_auto_json(self):
        data = [{"x": 1}, {"x": 2}]
        json_content = json.dumps(data).encode("utf-8")
        df, config = DataImporter.auto_import(json_content, "data.json")
        assert len(df) == 2

    def test_unsupported_format(self):
        with pytest.raises(ValueError, match="Unsupported"):
            DataImporter.auto_import(b"data", "file.xyz")