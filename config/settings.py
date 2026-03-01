"""
App-wide constants, defaults, and threshold configurations.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class Settings:
    """Immutable application settings."""

    # ── App Meta ──────────────────────────────────────────────
    APP_NAME: str = "DataPrep Kit"
    APP_VERSION: str = "0.1.1"
    APP_ICON: str = "🧪"

    # ── Import Defaults ───────────────────────────────────────
    MAX_UPLOAD_SIZE_MB: int = 500
    DEFAULT_CSV_DELIMITER: str = ","
    SUPPORTED_DELIMITERS: tuple = (",", ";", "\t", "|")
    SUPPORTED_ENCODINGS: tuple = (
        "utf-8", "latin-1", "iso-8859-1", "cp1252", "utf-16")
    SUPPORTED_FILE_TYPES: tuple = ("csv", "xlsx", "xls", "json", "parquet")
    PREVIEW_ROWS: int = 50
    SAMPLE_ROWS_DEFAULT: int = 10000

    # ── Missing Value Thresholds ──────────────────────────────
    MISSING_HIGH_THRESHOLD: float = 0.50  # >50% missing → suggest drop column
    MISSING_MEDIUM_THRESHOLD: float = 0.10  # >10% missing → suggest impute
    MISSING_LOW_THRESHOLD: float = 0.01  # >1% missing → flag but low priority

    # ── Outlier Defaults ──────────────────────────────────────
    IQR_MULTIPLIER: float = 1.5
    ZSCORE_THRESHOLD: float = 3.0
    PERCENTILE_LOWER: float = 0.01
    PERCENTILE_UPPER: float = 0.99

    # ── Duplicate Thresholds ──────────────────────────────────
    DUPLICATE_WARN_THRESHOLD: float = 0.01  # >1% duplicates → flag

    # ── Type Detection ────────────────────────────────────────
    TYPE_DETECTION_SAMPLE_SIZE: int = 1000
    # if unique/total < 5% → likely categorical
    CATEGORICAL_UNIQUE_RATIO: float = 0.05
    CATEGORICAL_MAX_UNIQUE: int = 50

    # ── Feature Engineering ───────────────────────────────────
    CORRELATION_HIGH_THRESHOLD: float = 0.95
    VARIANCE_LOW_THRESHOLD: float = 0.01
    RARE_CATEGORY_THRESHOLD: float = 0.01  # categories < 1% frequency → rare

    # ── Pipeline ──────────────────────────────────────────────
    MAX_UNDO_STEPS: int = 30
    PIPELINE_SCHEMA_VERSION: str = "1.0.0"

    # ── Export ────────────────────────────────────────────────
    EXPORT_FORMATS: tuple = ("csv", "xlsx", "json", "parquet")

    # ── AI Agent (Phase 2) ────────────────────────────────────
    AI_PROVIDERS: tuple = ("groq", "openai", "anthropic", "google")
    AI_DEFAULT_PROVIDER: str = "groq"
    AI_MODELS: dict = None  # type: ignore # set in __post_init__

    TARGET_TRACKS: tuple = (
        "Exploratory Data Analysis (EDA)",
        "Statistical Analysis",
        "Machine Learning (Classical ML)",
        "Deep Learning",
        "Time Series Analysis",
        "Natural Language Processing (NLP)",
        "Computer Vision",
        "Recommendation Systems",
        "Data Visualization / Reporting",
        "Data Warehousing / ETL",
    )

    AI_MAX_RECOMMENDATIONS_PER_DOMAIN: int = 10

    def __post_init__(self):
        # Bypass frozen for this one field
        object.__setattr__(self, "AI_MODELS", {
            "groq": ["llama-3.3-70b-versatile", "moonshotai/kimi-k2-instruct-0905", "openai/gpt-oss-120b"],
            "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1-nano"],
            "anthropic": ["claude-sonnet-4-20250514", "claude-3-5-haiku-20241022"],
            "google": ["gemini-2.0-flash", "gemini-2.5-flash-preview-05-20"],
        })


# Singleton instance
settings = Settings()
