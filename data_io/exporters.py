"""
Data exporters for CSV, Excel, JSON, and Parquet formats.
"""

import pandas as pd
import io
from typing import Optional


class DataExporter:
    """Handles exporting data to various file formats."""

    @staticmethod
    def to_csv(
        df: pd.DataFrame,
        index: bool = False,
        delimiter: str = ",",
        encoding: str = "utf-8",
    ) -> bytes:
        """Export DataFrame to CSV bytes."""
        buffer = io.StringIO()
        df.to_csv(buffer, index=index, sep=delimiter, encoding=encoding)
        return buffer.getvalue().encode(encoding)

    @staticmethod
    def to_excel(
        df: pd.DataFrame,
        sheet_name: str = "Sheet1",
        index: bool = False,
    ) -> bytes:
        """Export DataFrame to Excel bytes."""
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=index)
        return buffer.getvalue()

    @staticmethod
    def to_json(
        df: pd.DataFrame,
        orient: str = "records",
        indent: int = 2,
    ) -> bytes:
        """Export DataFrame to JSON bytes."""
        json_str = df.to_json(
            orient=orient,  # type: ignore
            indent=indent,
            default_handler=str
        )
        return json_str.encode("utf-8")  # type: ignore

    @staticmethod
    def to_parquet(
        df: pd.DataFrame,
        index: bool = False,
    ) -> bytes:
        """Export DataFrame to Parquet bytes."""
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=index)
        return buffer.getvalue()

    @classmethod
    def export(
        cls,
        df: pd.DataFrame,
        format: str,
        **kwargs,
    ) -> bytes:
        """Export DataFrame to the specified format."""
        exporters = {
            "csv": cls.to_csv,
            "xlsx": cls.to_excel,
            "json": cls.to_json,
            "parquet": cls.to_parquet,
        }

        exporter = exporters.get(format)
        if not exporter:
            raise ValueError(f"Unsupported export format: {format}")

        return exporter(df, **kwargs)

    @staticmethod
    def get_mime_type(format: str) -> str:
        """Get MIME type for download."""
        mime_types = {
            "csv": "text/csv",
            "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "json": "application/json",
            "parquet": "application/octet-stream",
        }
        return mime_types.get(format, "application/octet-stream")

    @staticmethod
    def get_file_extension(format: str) -> str:
        """Get proper file extension."""
        return format if format != "excel" else "xlsx"
