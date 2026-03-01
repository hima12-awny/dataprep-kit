"""
Regex-based and heuristic type inference for dataframe columns.
Detects: datetime, numeric-as-string, boolean-as-string, email, phone, etc.
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from config.settings import settings


class TypeDetector:
    """Analyzes columns and suggests appropriate data types."""

    # ── Regex Patterns ────────────────────────────────────────
    DATE_PATTERNS = [
        r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$",                    # 2024-01-15
        r"^\d{1,2}[-/]\d{1,2}[-/]\d{4}$",                    # 01/15/2024
        r"^\d{1,2}[-/]\d{1,2}[-/]\d{2}$",                    # 01/15/24
        r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}\s\d{1,2}:\d{2}",     # 2024-01-15 10:30
        # 15 Jan 2024
        r"^\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)",
        # Jan 15, 2024
        r"^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{1,2}",
        # Year only: 2024
        r"^\d{4}$",
    ]

    DATETIME_PATTERNS = [
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",            # ISO 8601
        # 2024-01-15 10:30:00
        r"^\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}",
    ]

    BOOLEAN_VALUES = {
        "true", "false", "yes", "no", "1", "0",
        "t", "f", "y", "n", "on", "off"
    }

    NUMERIC_PATTERN = r"^-?\d+\.?\d*$"
    INTEGER_PATTERN = r"^-?\d+$"
    EMAIL_PATTERN = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    PHONE_PATTERN = r"^[\+]?[(]?[0-9]{1,4}[)]?[-\s\./0-9]*$"
    URL_PATTERN = r"^https?://[^\s]+$"

    @classmethod
    def analyze_column(cls, series: pd.Series) -> Dict:
        """
        Analyze a single column and return type detection results.

        Returns:
            {
                "current_type": str,
                "suggested_type": str | None,
                "confidence": float,
                "reason": str,
                "detected_pattern": str | None,
                "sample_values": list
            }
        """
        current_type = str(series.dtype)
        non_null = series.dropna()

        if len(non_null) == 0:
            return {
                "current_type": current_type,
                "suggested_type": None,
                "confidence": 0.0,
                "reason": "Column is entirely null",
                "detected_pattern": None,
                "sample_values": [],
            }

        # Sample for performance
        sample_size = min(len(non_null), settings.TYPE_DETECTION_SAMPLE_SIZE)
        sample = non_null.sample(n=sample_size, random_state=42) if len(
            non_null) > sample_size else non_null

        result = {
            "current_type": current_type,
            "suggested_type": None,
            "confidence": 0.0,
            "reason": "No type change needed",
            "detected_pattern": None,
            "sample_values": non_null.head(5).tolist(),
        }

        # Only suggest changes for object/string columns
        if series.dtype == "object":
            checks = [
                cls._check_boolean,
                cls._check_datetime,
                cls._check_integer,
                cls._check_numeric,
                cls._check_categorical,
            ]
            for check in checks:
                detection = check(sample)
                if detection and detection["confidence"] > result["confidence"]:
                    result.update(detection)

        # Check if numeric could be categorical
        elif pd.api.types.is_numeric_dtype(series):
            cat_check = cls._check_numeric_as_categorical(sample, series)
            if cat_check:
                result.update(cat_check)

        return result

    @classmethod
    def analyze_dataframe(cls, df: pd.DataFrame) -> Dict[str, Dict]:
        """Analyze all columns in a dataframe."""
        results = {}
        for col in df.columns:
            results[col] = cls.analyze_column(df[col])  # type: ignore
        return results

    @classmethod
    def get_suggested_conversions(cls, df: pd.DataFrame) -> List[Dict]:
        """Return only columns where a type change is suggested."""
        analysis = cls.analyze_dataframe(df)
        suggestions = []
        for col, result in analysis.items():
            if result["suggested_type"] and result["confidence"] >= 0.7:
                suggestions.append({"column": col, **result})
        return suggestions

    # ── Private Detection Methods ─────────────────────────────

    @classmethod
    def _check_datetime(cls, sample: pd.Series) -> Optional[Dict]:
        str_values = sample.astype(str).str.strip()

        # Try pandas date parsing (without deprecated infer_datetime_format)
        try:
            parsed = pd.to_datetime(str_values, format="mixed", dayfirst=False)
            success_rate = parsed.notna().mean()
            if success_rate >= 0.8:
                return {
                    "suggested_type": "datetime64[ns]",
                    "confidence": round(success_rate, 2),
                    "reason": f"{success_rate:.0%} of values parse as datetime",
                    "detected_pattern": "auto-detected",
                }
        except (ValueError, TypeError):
            pass

        # Try regex patterns
        for pattern in cls.DATE_PATTERNS + cls.DATETIME_PATTERNS:
            match_rate = str_values.str.match(
                pattern, case=False, na=False).mean()
            if match_rate >= 0.7:
                return {
                    "suggested_type": "datetime64[ns]",
                    "confidence": round(match_rate, 2),
                    "reason": f"{match_rate:.0%} of values match date pattern",
                    "detected_pattern": pattern,
                }
        return None

    @classmethod
    def _check_boolean(cls, sample: pd.Series) -> Optional[Dict]:
        str_values = sample.astype(str).str.strip().str.lower()
        bool_rate = str_values.isin(cls.BOOLEAN_VALUES).mean()
        if bool_rate >= 0.9:
            return {
                "suggested_type": "bool",
                "confidence": round(bool_rate, 2),
                "reason": f"{bool_rate:.0%} of values are boolean-like",
                "detected_pattern": "boolean_values",
            }
        return None

    @classmethod
    def _check_integer(cls, sample: pd.Series) -> Optional[Dict]:
        str_values = sample.astype(str).str.strip()
        match_rate = str_values.str.match(cls.INTEGER_PATTERN, na=False).mean()
        if match_rate >= 0.9:
            return {
                "suggested_type": "int64",
                "confidence": round(match_rate, 2),
                "reason": f"{match_rate:.0%} of values are integers stored as strings",
                "detected_pattern": "integer",
            }
        return None

    @classmethod
    def _check_numeric(cls, sample: pd.Series) -> Optional[Dict]:
        str_values = sample.astype(str).str.strip()
        match_rate = str_values.str.match(cls.NUMERIC_PATTERN, na=False).mean()
        if match_rate >= 0.85:
            return {
                "suggested_type": "float64",
                "confidence": round(match_rate, 2),
                "reason": f"{match_rate:.0%} of values are numeric stored as strings",
                "detected_pattern": "numeric",
            }
        return None

    @classmethod
    def _check_categorical(cls, sample: pd.Series) -> Optional[Dict]:
        n_unique = sample.nunique()
        total = len(sample)
        ratio = n_unique / total if total > 0 else 1.0

        if ratio <= settings.CATEGORICAL_UNIQUE_RATIO and n_unique <= settings.CATEGORICAL_MAX_UNIQUE:
            return {
                "suggested_type": "category",
                "confidence": round(1 - ratio, 2),
                "reason": f"Only {n_unique} unique values ({ratio:.1%} of rows) — likely categorical",
                "detected_pattern": "low_cardinality",
            }
        return None

    @classmethod
    def _check_numeric_as_categorical(cls, sample: pd.Series, full_series: pd.Series) -> Optional[Dict]:
        n_unique = full_series.nunique()
        total = len(full_series.dropna())
        ratio = n_unique / total if total > 0 else 1.0

        if ratio <= settings.CATEGORICAL_UNIQUE_RATIO and n_unique <= settings.CATEGORICAL_MAX_UNIQUE:
            return {
                "suggested_type": "category",
                "confidence": round(1 - ratio, 2),
                "reason": f"Numeric column with only {n_unique} unique values — may be categorical",
                "detected_pattern": "numeric_low_cardinality",
            }
        return None
