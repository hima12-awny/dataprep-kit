"""
Data importers for CSV, Excel, JSON, and Parquet files.
"""

import pandas as pd
import io
from typing import Dict, Optional, Any, Tuple, List

from config.settings import settings


class DataImporter:
    """Handles importing data from various file formats."""

    @staticmethod
    def import_csv(
        file_content,
        delimiter: str = ",",
        encoding: str = "utf-8",
        header_row: Optional[int] = 0,
        index_col: Optional[int] = None,
        sample_rows: Optional[int] = None,
        na_values: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Import a CSV file.

        Returns:
            (DataFrame, import_config dict)
        """
        kwargs = {
            "sep": delimiter,
            "encoding": encoding,
            "header": header_row,
            "index_col": index_col,
            "na_values": na_values or ["", "NA", "N/A", "null", "NULL", "None", "nan", "NaN"],
        }

        if sample_rows:
            kwargs["nrows"] = sample_rows

        if isinstance(file_content, bytes):
            file_content = io.BytesIO(file_content)

        df = pd.read_csv(file_content, **kwargs)

        config = {
            "format": "csv",
            "delimiter": delimiter,
            "encoding": encoding,
            "header_row": header_row,
            "index_col": index_col,
            "sample_rows": sample_rows,
        }

        return df, config

    @staticmethod
    def import_excel(
        file_content,
        sheet_name: Any = 0,
        header_row: Optional[int] = 0,
        index_col: Optional[int] = None,
        sample_rows: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """Import an Excel file."""
        kwargs = {
            "sheet_name": sheet_name,
            "header": header_row,
            "index_col": index_col,
        }

        if sample_rows:
            kwargs["nrows"] = sample_rows

        if isinstance(file_content, bytes):
            file_content = io.BytesIO(file_content)

        df = pd.read_excel(file_content, **kwargs)

        config = {
            "format": "excel",
            "sheet_name": sheet_name,
            "header_row": header_row,
            "index_col": index_col,
            "sample_rows": sample_rows,
        }

        return df, config

    @staticmethod
    def import_json(
        file_content,
        orient: str = "records",
        lines: bool = False,
    ) -> Tuple[pd.DataFrame, Dict]:
        """Import a JSON file."""
        if isinstance(file_content, bytes):
            file_content = io.BytesIO(file_content)

        df = pd.read_json(file_content, orient=orient, lines=lines)

        config = {
            "format": "json",
            "orient": orient,
            "lines": lines,
        }

        return df, config

    @staticmethod
    def import_parquet(file_content) -> Tuple[pd.DataFrame, Dict]:
        """Import a Parquet file."""
        if isinstance(file_content, bytes):
            file_content = io.BytesIO(file_content)

        df = pd.read_parquet(file_content)
        config = {"format": "parquet"}
        return df, config

    @classmethod
    def auto_import(
        cls,
        file_content,
        filename: str,
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Auto-detect format from filename extension and import accordingly.
        """
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

        if ext == "csv":
            return cls.import_csv(file_content, **kwargs)
        elif ext in ("xlsx", "xls"):
            return cls.import_excel(file_content, **kwargs)
        elif ext == "json":
            return cls.import_json(file_content, **kwargs)
        elif ext == "parquet":
            return cls.import_parquet(file_content)
        else:
            raise ValueError(f"Unsupported file format: .{ext}")

    @staticmethod
    def get_excel_sheet_names(file_content) -> List[str]:
        """Get sheet names from an Excel file."""
        if isinstance(file_content, bytes):
            file_content = io.BytesIO(file_content)
        xls = pd.ExcelFile(file_content)
        return xls.sheet_names

    @staticmethod
    def detect_csv_delimiter(file_content, n_lines: int = 10) -> str:
        """Attempt to detect the CSV delimiter."""
        import csv

        if isinstance(file_content, bytes):
            sample_text = file_content[:8192].decode("utf-8", errors="ignore")
        else:
            sample_text = file_content.read(8192)
            file_content.seek(0)

        try:
            dialect = csv.Sniffer().sniff(sample_text)
            return dialect.delimiter
        except csv.Error:
            return ","

    @staticmethod
    def detect_encoding(file_content) -> str:
        """Attempt to detect file encoding."""
        try:
            import chardet
            raw = file_content[:10000] if isinstance(
                file_content, bytes) else file_content.read(10000)
            if not isinstance(raw, bytes):
                return "utf-8"
            result = chardet.detect(raw)
            return result.get("encoding", "utf-8") or "utf-8"
        except ImportError:
            return "utf-8"
