"""
Statistical helper functions for data profiling, outlier detection,
and distribution analysis. All methods handle unhashable types safely.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any


class StatsHelper:
    """
    Central statistics engine for DataPrep Kit.
    Provides safe, robust methods for profiling, outlier detection,
    correlation analysis, and distribution summaries.
    """

    # ══════════════════════════════════════════════════════════
    #  DataFrame Overview
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def dataframe_overview(df: pd.DataFrame) -> Dict:
        """
        Generate a comprehensive overview of the dataframe.

        Returns:
            Dict with row_count, column_count, total_cells,
            total_missing, missing_percentage, duplicate_rows,
            duplicate_percentage, memory_usage_mb.
        """
        row_count = len(df)
        col_count = len(df.columns)
        total_cells = row_count * col_count
        total_missing = int(df.isna().sum().sum())
        missing_pct = round((total_missing / total_cells * 100),
                            2) if total_cells > 0 else 0.0

        duplicate_rows = StatsHelper.safe_duplicate_count(df)
        duplicate_pct = round(
            (duplicate_rows / row_count * 100), 2) if row_count > 0 else 0.0

        memory_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)

        return {
            "row_count": row_count,
            "column_count": col_count,
            "total_cells": total_cells,
            "total_missing": total_missing,
            "missing_percentage": missing_pct,
            "duplicate_rows": duplicate_rows,
            "duplicate_percentage": duplicate_pct,
            "memory_usage_mb": memory_mb,
        }

    # ══════════════════════════════════════════════════════════
    #  Column Statistics
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def get_column_stats(df: pd.DataFrame, col_name: str) -> Dict:
        """
        Generate detailed statistics for a single column.

        Returns:
            Dict with basic info, type-specific stats, and most common values.
        """
        if col_name not in df.columns:
            return {"error": f"Column '{col_name}' not found"}

        series = df[col_name]
        total = len(series)
        non_null = int(series.notna().sum())
        null_count = int(series.isna().sum())
        null_pct = round((null_count / total * 100), 2) if total > 0 else 0.0
        unique = StatsHelper.safe_nunique(series)
        unique_pct = round((unique / total * 100), 2) if total > 0 else 0.0

        stats = {
            "column_name": col_name,
            "dtype": str(series.dtype),
            "total_count": total,
            "non_null_count": non_null,
            "null_count": null_count,
            "null_percentage": null_pct,
            "unique_count": unique,
            "unique_percentage": unique_pct,
        }

        # ── Numeric stats ─────────────────────────────────
        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            if len(clean) > 0:
                stats.update({
                    "mean": round(float(clean.mean()), 4),
                    "median": round(float(clean.median()), 4),
                    "std": round(float(clean.std()), 4),
                    "min": round(float(clean.min()), 4),
                    "max": round(float(clean.max()), 4),
                    "q1": round(float(clean.quantile(0.25)), 4),
                    "q3": round(float(clean.quantile(0.75)), 4),
                    "iqr": round(float(clean.quantile(0.75) - clean.quantile(0.25)), 4),
                    "skewness": round(float(clean.skew()), 4),
                    "kurtosis": round(float(clean.kurtosis()), 4),
                    "zeros": int((clean == 0).sum()),
                    "negatives": int((clean < 0).sum()),
                })

        # ── Datetime stats ────────────────────────────────
        elif pd.api.types.is_datetime64_any_dtype(series):
            clean = series.dropna()
            if len(clean) > 0:
                min_date = clean.min()
                max_date = clean.max()
                stats.update({
                    "min": str(min_date),
                    "max": str(max_date),
                    "date_range": str(max_date - min_date),
                })

        # ── Text stats ────────────────────────────────────
        elif series.dtype in ("object", "string") or str(series.dtype) == "object":
            clean = series.dropna()
            if len(clean) > 0:
                # Safe string length computation
                try:
                    lengths = clean.astype(str).str.len()
                    stats.update({
                        "avg_length": round(float(lengths.mean()), 2),
                        "min_length": int(lengths.min()),
                        "max_length": int(lengths.max()),
                    })
                except Exception:
                    pass

                # Whitespace detection
                try:
                    str_vals = clean.astype(str)
                    has_ws = bool((str_vals != str_vals.str.strip()).any())
                    stats["has_whitespace_issues"] = has_ws
                except Exception:
                    stats["has_whitespace_issues"] = False

                # Mixed case detection
                try:
                    str_vals = clean.astype(str)
                    lowered = str_vals.str.lower()
                    unique_original = StatsHelper.safe_nunique(str_vals)
                    unique_lowered = StatsHelper.safe_nunique(lowered)
                    stats["has_mixed_case"] = unique_lowered < unique_original
                except Exception:
                    stats["has_mixed_case"] = False

        # ── Most common values (all types) ────────────────
        stats["most_common"] = StatsHelper.safe_value_counts(series, 10)

        return stats

    # ══════════════════════════════════════════════════════════
    #  Correlation Analysis
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> Optional[pd.DataFrame]:
        """
        Compute correlation matrix for numeric columns.

        Args:
            df: Input dataframe.
            method: Correlation method ('pearson', 'spearman', 'kendall').

        Returns:
            Correlation DataFrame or None if fewer than 2 numeric columns.
        """
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            return None

        try:
            return numeric_df.corr(method=method).round(4)
        except Exception:
            return None

    @staticmethod
    def high_correlation_pairs(
        df: pd.DataFrame,
        threshold: float = 0.95,
        method: str = "pearson",
    ) -> List[Tuple[str, str, float]]:
        """
        Find pairs of columns with correlation above the threshold.

        Returns:
            List of (col1, col2, correlation_value) tuples sorted by |corr| descending.
        """
        corr = StatsHelper.correlation_matrix(df, method=method)
        if corr is None:
            return []

        pairs = []
        seen = set()
        for col1 in corr.columns:
            for col2 in corr.columns:
                if col1 == col2:
                    continue
                pair_key = tuple(sorted([col1, col2]))
                if pair_key in seen:
                    continue

                corr_val = corr.loc[col1, col2]
                if abs(corr_val) >= threshold:
                    pairs.append((col1, col2, round(float(corr_val), 4)))
                    seen.add(pair_key)

        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        return pairs

    # ══════════════════════════════════════════════════════════
    #  Outlier Detection
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def iqr_outliers(
        series: pd.Series, multiplier: float = 1.5
    ) -> Dict:
        """
        Detect outliers using the IQR method.

        Returns:
            Dict with lower_bound, upper_bound, outlier_count,
            outlier_percentage, and outlier_indices.
        """
        clean = series.dropna()
        if len(clean) == 0 or not pd.api.types.is_numeric_dtype(clean):
            return {"outlier_count": 0, "outlier_percentage": 0.0, "outlier_indices": []}

        q1 = float(clean.quantile(0.25))
        q3 = float(clean.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr

        outlier_mask = (clean < lower) | (clean > upper)
        outlier_count = int(outlier_mask.sum())
        outlier_pct = round((outlier_count / len(clean) * 100),
                            2) if len(clean) > 0 else 0.0

        return {
            "lower_bound": round(lower, 4),
            "upper_bound": round(upper, 4),
            "outlier_count": outlier_count,
            "outlier_percentage": outlier_pct,
            "outlier_indices": clean[outlier_mask].index.tolist(),
        }

    @staticmethod
    def zscore_outliers(
        series: pd.Series, threshold: float = 3.0
    ) -> Dict:
        """
        Detect outliers using the Z-Score method.

        Returns:
            Dict with outlier_count, outlier_percentage, and outlier_indices.
        """
        clean = series.dropna()
        if len(clean) == 0 or not pd.api.types.is_numeric_dtype(clean):
            return {"outlier_count": 0, "outlier_percentage": 0.0, "outlier_indices": []}

        mean = float(clean.mean())
        std = float(clean.std())

        if std == 0:
            return {"outlier_count": 0, "outlier_percentage": 0.0, "outlier_indices": []}

        zscores = ((clean - mean) / std).abs()
        outlier_mask = zscores > threshold
        outlier_count = int(outlier_mask.sum())
        outlier_pct = round((outlier_count / len(clean) * 100),
                            2) if len(clean) > 0 else 0.0

        return {
            "threshold": threshold,
            "outlier_count": outlier_count,
            "outlier_percentage": outlier_pct,
            "outlier_indices": clean[outlier_mask].index.tolist(),
        }

    @staticmethod
    def percentile_outliers(
        series: pd.Series,
        lower_pct: float = 0.01,
        upper_pct: float = 0.99,
    ) -> Dict:
        """
        Detect outliers using percentile bounds.

        Returns:
            Dict with lower_bound, upper_bound, outlier_count,
            outlier_percentage, and outlier_indices.
        """
        clean = series.dropna()
        if len(clean) == 0 or not pd.api.types.is_numeric_dtype(clean):
            return {"outlier_count": 0, "outlier_percentage": 0.0, "outlier_indices": []}

        lower = float(clean.quantile(lower_pct))
        upper = float(clean.quantile(upper_pct))

        outlier_mask = (clean < lower) | (clean > upper)
        outlier_count = int(outlier_mask.sum())
        outlier_pct = round((outlier_count / len(clean) * 100),
                            2) if len(clean) > 0 else 0.0

        return {
            "lower_bound": round(lower, 4),
            "upper_bound": round(upper, 4),
            "outlier_count": outlier_count,
            "outlier_percentage": outlier_pct,
            "outlier_indices": clean[outlier_mask].index.tolist(),
        }

    @staticmethod
    def detect_outliers(
        series: pd.Series,
        method: str = "iqr",
        threshold: float = 1.5,
    ) -> Dict:
        """
        Unified outlier detection dispatcher.

        Args:
            series: Numeric series.
            method: 'iqr', 'zscore', or 'percentile'.
            threshold: Method-specific threshold.

        Returns:
            Outlier detection results dict.
        """
        if method == "iqr":
            return StatsHelper.iqr_outliers(series, multiplier=threshold)
        elif method == "zscore":
            return StatsHelper.zscore_outliers(series, threshold=threshold)
        elif method == "percentile":
            return StatsHelper.percentile_outliers(series)
        else:
            raise ValueError(f"Unknown outlier method: {method}")

    # ══════════════════════════════════════════════════════════
    #  Distribution Analysis
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def skewness(series: pd.Series) -> Optional[float]:
        """Calculate skewness of a numeric series."""
        clean = series.dropna()
        if len(clean) < 3 or not pd.api.types.is_numeric_dtype(clean):
            return None
        try:
            return round(float(clean.skew()), 4)
        except Exception:
            return None

    @staticmethod
    def kurtosis(series: pd.Series) -> Optional[float]:
        """Calculate kurtosis of a numeric series."""
        clean = series.dropna()
        if len(clean) < 4 or not pd.api.types.is_numeric_dtype(clean):
            return None
        try:
            return round(float(clean.kurtosis()), 4)
        except Exception:
            return None

    @staticmethod
    def is_skewed(series: pd.Series, threshold: float = 1.0) -> bool:
        """Check if a series is significantly skewed."""
        skew = StatsHelper.skewness(series)
        if skew is None:
            return False
        return abs(skew) > threshold

    @staticmethod
    def distribution_type(series: pd.Series) -> str:
        """
        Classify the distribution shape of a numeric series.

        Returns:
            One of: 'symmetric', 'right_skewed', 'left_skewed',
            'heavily_right_skewed', 'heavily_left_skewed', 'unknown'.
        """
        skew = StatsHelper.skewness(series)
        if skew is None:
            return "unknown"

        if abs(skew) < 0.5:
            return "symmetric"
        elif skew > 0:
            return "right_skewed" if skew < 2 else "heavily_right_skewed"
        else:
            return "left_skewed" if skew > -2 else "heavily_left_skewed"

    # ══════════════════════════════════════════════════════════
    #  Variance & Cardinality
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def low_variance_columns(
        df: pd.DataFrame, threshold: float = 0.01
    ) -> List[str]:
        """Find numeric columns with variance below threshold."""
        low_var = []
        for col in df.select_dtypes(include=[np.number]).columns:
            clean = df[col].dropna()
            if len(clean) > 0:
                var = float(clean.var())
                if var < threshold:
                    low_var.append(col)
        return low_var

    @staticmethod
    def constant_columns(df: pd.DataFrame) -> List[str]:
        """Find columns with only one unique value (or all nulls)."""
        constants = []
        for col in df.columns:
            nunique = StatsHelper.safe_nunique(df[col])
            if nunique <= 1:
                constants.append(col)
        return constants

    @staticmethod
    def high_cardinality_columns(
        df: pd.DataFrame, threshold: float = 0.9
    ) -> List[str]:
        """
        Find columns where unique values / total rows > threshold.
        These are likely identifiers.
        """
        high_card = []
        total = len(df)
        if total == 0:
            return []

        for col in df.columns:
            nunique = StatsHelper.safe_nunique(df[col])
            if nunique / total > threshold:
                high_card.append(col)
        return high_card

    @staticmethod
    def rare_categories(
        series: pd.Series, threshold: float = 0.01
    ) -> List[str]:
        """
        Find category values that appear less than threshold frequency.

        Args:
            series: Categorical/object series.
            threshold: Minimum frequency (0.01 = 1%).

        Returns:
            List of rare category values as strings.
        """
        try:
            vc = series.value_counts(normalize=True)
            rare = vc[vc < threshold].index.tolist()
            return [str(v) for v in rare]
        except TypeError:
            try:
                vc = series.astype(str).value_counts(normalize=True)
                rare = vc[vc < threshold].index.tolist()
                return [str(v) for v in rare]
            except Exception:
                return []

    # ══════════════════════════════════════════════════════════
    #  Safe Methods (handle unhashable types)
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def safe_duplicate_count(df: pd.DataFrame) -> int:
        """
        Count duplicate rows safely, handling columns with unhashable
        types (lists, dicts, numpy arrays, etc.).
        """
        try:
            return int(df.duplicated().sum())
        except (TypeError, ValueError):
            try:
                hashable_df = df.copy()
                for col in hashable_df.columns:
                    if hashable_df[col].dtype == "object":
                        sample = hashable_df[col].dropna().head(100)
                        has_unhashable = False
                        for val in sample:
                            try:
                                hash(val)
                            except TypeError:
                                has_unhashable = True
                                break
                        if has_unhashable:
                            hashable_df[col] = hashable_df[col].astype(str)
                return int(hashable_df.duplicated().sum())
            except Exception:
                return 0

    @staticmethod
    def safe_nunique(series: pd.Series) -> int:
        """Count unique values safely, handling unhashable types."""
        try:
            return int(series.nunique())
        except TypeError:
            try:
                # Try converting lists to tuples for better accuracy than conversion to string
                return int(StatsHelper.ensure_hashable(series).nunique())
            except Exception:
                try:
                    return int(series.astype(str).nunique())
                except Exception:
                    return 0

    @staticmethod
    def ensure_hashable(series: pd.Series) -> pd.Series:
        """
        Convert unhashable values in a series (like lists) to hashable ones (tuples).
        """
        return series.apply(StatsHelper.recursive_tuple)

    @staticmethod
    def recursive_tuple(val: Any) -> Any:
        """Recursively convert lists to tuples."""
        if isinstance(val, list):
            return tuple(StatsHelper.recursive_tuple(item) for item in val)
        if isinstance(val, dict):
            return tuple(sorted((k, StatsHelper.recursive_tuple(v)) for k, v in val.items()))
        if isinstance(val, np.ndarray):
            return tuple(val.tolist())
        return val

    @staticmethod
    def safe_value_counts(series: pd.Series, n: int = 10) -> Dict[str, int]:
        """Get value counts safely, handling unhashable types."""
        try:
            vc = series.value_counts().head(n)
            return {str(k): int(v) for k, v in vc.items()}
        except TypeError:
            try:
                vc = series.astype(str).value_counts().head(n)
                return {str(k): int(v) for k, v in vc.items()}
            except Exception:
                return {}

    @staticmethod
    def safe_duplicated_mask(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.Series:
        """
        Get a boolean mask of duplicated rows, safe for unhashable columns.

        Args:
            df: Input dataframe.
            subset: Columns to check. None = all.

        Returns:
            Boolean Series indicating duplicate rows.
        """
        check_df = df[subset] if subset else df
        try:
            return check_df.duplicated(keep=False)
        except (TypeError, ValueError):
            try:
                hashable = check_df.copy()
                for col in hashable.columns:
                    if hashable[col].dtype == "object":
                        sample = hashable[col].dropna().head(50)
                        for val in sample:
                            try:
                                hash(val)
                            except TypeError:
                                hashable[col] = hashable[col].astype(str)
                                break
                return hashable.duplicated(keep=False)
            except Exception:
                return pd.Series([False] * len(df), index=df.index)

    # ══════════════════════════════════════════════════════════
    #  Numeric Helpers
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a summary table for all numeric columns.

        Returns:
            DataFrame with mean, median, std, min, max, skew, kurtosis per column.
        """
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return pd.DataFrame()

        rows = []
        for col in numeric_df.columns:
            clean = numeric_df[col].dropna()
            if len(clean) == 0:
                continue
            rows.append({
                "Column": col,
                "Mean": round(float(clean.mean()), 4),
                "Median": round(float(clean.median()), 4),
                "Std": round(float(clean.std()), 4),
                "Min": round(float(clean.min()), 4),
                "Max": round(float(clean.max()), 4),
                "Skewness": round(float(clean.skew()), 4),
                "Kurtosis": round(float(clean.kurtosis()), 4),
                "Nulls": int(numeric_df[col].isna().sum()),
            })

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    @staticmethod
    def categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a summary table for all categorical/object columns.

        Returns:
            DataFrame with unique count, top value, frequency, and null count.
        """
        cat_df = df.select_dtypes(include=["object", "category"])
        if cat_df.empty:
            return pd.DataFrame()

        rows = []
        for col in cat_df.columns:
            series = cat_df[col]
            nunique = StatsHelper.safe_nunique(series)
            top_values = StatsHelper.safe_value_counts(series, 1)
            top_val = list(top_values.keys())[0] if top_values else "N/A"
            top_freq = list(top_values.values())[0] if top_values else 0

            rows.append({
                "Column": col,
                "Unique": nunique,
                "Top Value": str(top_val),
                "Top Frequency": top_freq,
                "Nulls": int(series.isna().sum()),
                "Null %": round(series.isna().mean() * 100, 2),
            })

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    # ══════════════════════════════════════════════════════════
    #  Imputation Strategy Suggestion
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def suggest_imputation(series: pd.Series) -> str:
        """
        Suggest an imputation strategy for a series based on its distribution.

        Returns:
            One of: 'mean', 'median', 'mode', 'drop_column', 'constant'.
        """
        null_pct = series.isna().mean()

        # Very high missing → drop
        if null_pct > 0.5:
            return "drop_column"

        # Numeric column
        if pd.api.types.is_numeric_dtype(series):
            skew = StatsHelper.skewness(series)
            if skew is not None and abs(skew) > 1.0:
                return "median"  # Robust to skewness
            return "mean"

        # Categorical / text
        return "mode"

    @staticmethod
    def suggest_outlier_behavior(
        series: pd.Series, method: str = "iqr", threshold: float = 1.5
    ) -> str:
        """
        Suggest outlier handling behavior based on outlier count.

        Returns:
            One of: 'clip', 'flag', 'remove'.
        """
        result = StatsHelper.detect_outliers(
            series, method=method, threshold=threshold)
        pct = result.get("outlier_percentage", 0)

        if pct > 10:
            return "clip"       # Too many to remove
        elif pct > 2:
            return "flag"       # Moderate — review needed
        elif pct > 0:
            return "remove"     # Few and likely erroneous
        return "flag"           # Default safe option
