"""
Pydantic models for AI agent structured outputs.
Domain-aware action schemas ensure each domain only receives valid action types.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Literal, Union
from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════
#  Data Description Models
# ══════════════════════════════════════════════════════════════

class ColumnInfo(BaseModel):
    """Description of a single column."""
    name: str = Field(description="Column name")
    description: str = Field(description="What this column represents")
    suggested_role: Literal["feature", "target", "identifier", "metadata", "drop"] = Field(
        description="Suggested role in modeling"
    )
    data_quality: Literal["good", "needs_attention", "poor"] = Field(
        description="Quality assessment"
    )
    quality_notes: str = Field(default="", description="Specific quality observations")


class DatasetDescription(BaseModel):
    """Structured description of the entire dataset."""
    summary: str = Field(description="2-3 sentence summary of what this dataset contains")
    domain: str = Field(
        description="Domain/industry this data belongs to (e.g., healthcare, finance, e-commerce)"
    )
    row_description: str = Field(
        description="What each row represents (e.g., 'one customer transaction')"
    )
    column_descriptions: List[ColumnInfo] = Field(description="Description of each column")
    data_quality_notes: List[str] = Field(description="Overall data quality observations")
    potential_issues: List[str] = Field(description="Potential problems that need attention")
    recommended_target_columns: List[str] = Field(
        description="Columns that could serve as prediction targets"
    )


# ══════════════════════════════════════════════════════════════
#  Domain-Aware Action Recommendation Models
# ══════════════════════════════════════════════════════════════

# ── Cleaning Domain ───────────────────────────────────────────

class CleaningActionParameters(BaseModel):
    """Parameters for a cleaning action. Fields are optional because different action types need different params."""
    columns: Optional[List[str]] = Field(default=None, description="Columns to process")
    strategy: Optional[str] = Field(default=None, description="Strategy for handle_missing")
    fill_value: Optional[Any] = Field(default=None, description="Fill value for constant strategy")
    group_by: Optional[str] = Field(default=None, description="Group by column for group_based strategy")
    group_strategy: Optional[str] = Field(default=None, description="Aggregation within groups")
    subset: Optional[List[str]] = Field(default=None, description="Subset columns for duplicates")
    keep: Optional[str] = Field(default=None, description="Which duplicate to keep")
    method: Optional[str] = Field(default=None, description="Outlier detection method")
    threshold: Optional[float] = Field(default=None, description="Threshold value")
    behavior: Optional[str] = Field(default=None, description="Outlier handling behavior")
    operations: Optional[List[str]] = Field(default=None, description="Text cleaning operations")
    operation: Optional[str] = Field(default=None, description="Inconsistency operation")
    mapping: Optional[Dict[str, str]] = Field(default=None, description="Value mapping dict")
    replacement: Optional[str] = Field(default=None, description="Replacement for rare categories")


class CleaningActionRecommendation(BaseModel):
    """A single cleaning action recommendation."""
    action_type: Literal[
        "handle_missing",
        "handle_duplicates",
        "handle_outliers",
        "text_cleaning",
        "inconsistency",
    ] = Field(description="Cleaning action type")
    description: str = Field(description="Human-readable description of what this action does")
    reason: str = Field(description="Why this action is recommended for this dataset")
    priority: Literal["high", "medium", "low"] = Field(description="Action priority")
    parameters: CleaningActionParameters = Field(description="Action parameters")


class CleaningRecommendationList(BaseModel):
    """List of cleaning action recommendations from the AI agent."""
    reasoning: str = Field(
        description="Brief overall analysis of the dataset's cleaning needs"
    )
    recommendations: List[CleaningActionRecommendation] = Field(
        description="Ordered list of recommended cleaning actions"
    )


# ── Conversion Domain ─────────────────────────────────────────

class ConversionActionParameters(BaseModel):
    """Parameters for a conversion action."""
    columns: Optional[List[str]] = Field(default=None, description="Columns to process")
    conversions: Optional[Dict[str, str]] = Field(default=None, description="Column-to-type mapping")
    errors: Optional[str] = Field(default=None, description="Error handling strategy")
    date_format: Optional[str] = Field(default=None, description="Datetime format string")
    operation: Optional[str] = Field(default=None, description="Operation type")
    column: Optional[str] = Field(default=None, description="Single source column")
    column2: Optional[str] = Field(default=None, description="Second column for date_diff")
    components: Optional[List[str]] = Field(default=None, description="Datetime components")
    unit: Optional[str] = Field(default=None, description="Unit for date_diff")
    overwrite: Optional[bool] = Field(default=None, description="Overwrite original column")
    method: Optional[str] = Field(default=None, description="Encoding method")
    drop_first: Optional[bool] = Field(default=None, description="Drop first dummy")
    drop_original: Optional[bool] = Field(default=None, description="Drop original after encoding")
    target_column: Optional[str] = Field(default=None, description="Target column for target encoding")
    n_bins: Optional[int] = Field(default=None, description="Number of bins for binning")


class ConversionActionRecommendation(BaseModel):
    """A single conversion action recommendation."""
    action_type: Literal[
        "type_casting",
        "datetime_ops",
        "numeric_transform",
        "encoding",
    ] = Field(description="Conversion action type")
    description: str = Field(description="Human-readable description of what this action does")
    reason: str = Field(description="Why this action is recommended for this dataset")
    priority: Literal["high", "medium", "low"] = Field(description="Action priority")
    parameters: ConversionActionParameters = Field(description="Action parameters")


class ConversionRecommendationList(BaseModel):
    """List of conversion action recommendations from the AI agent."""
    reasoning: str = Field(
        description="Brief overall analysis of the dataset's conversion needs"
    )
    recommendations: List[ConversionActionRecommendation] = Field(
        description="Ordered list of recommended conversion actions"
    )


# ── Feature Engineering Domain ────────────────────────────────

class EngineeringActionParameters(BaseModel):
    """Parameters for a feature engineering action."""
    columns: Optional[List[str]] = Field(default=None, description="Columns to process")
    column: Optional[str] = Field(default=None, description="Single source column")
    operation: Optional[str] = Field(default=None, description="Operation type")
    new_column: Optional[str] = Field(default=None, description="Name for new column")
    expression: Optional[str] = Field(default=None, description="Python expression")
    separator: Optional[str] = Field(default=None, description="Separator for combine/cross")
    delimiter: Optional[str] = Field(default=None, description="Delimiter for split")
    rename_map: Optional[Dict[str, str]] = Field(default=None, description="Rename mapping")
    group_by: Optional[Union[str, List[str]]] = Field(default=None, description="Group by column(s)")
    agg_columns: Optional[List[str]] = Field(default=None, description="Columns to aggregate")
    agg_func: Optional[str] = Field(default=None, description="Aggregation function")
    merge_back: Optional[bool] = Field(default=None, description="Merge aggregated back")
    periods: Optional[int] = Field(default=None, description="Lag/lead periods")
    window: Optional[int] = Field(default=None, description="Rolling window size")
    rolling_func: Optional[str] = Field(default=None, description="Rolling function")
    max_value: Optional[float] = Field(default=None, description="Max cycle value for cyclical")
    degree: Optional[int] = Field(default=None, description="Polynomial degree")
    drop_original: Optional[bool] = Field(default=None, description="Drop original column")


class EngineeringActionRecommendation(BaseModel):
    """A single feature engineering action recommendation."""
    action_type: Literal[
        "column_ops",
        "aggregation",
        "temporal",
        "interaction",
    ] = Field(description="Feature engineering action type")
    description: str = Field(description="Human-readable description of what this action does")
    reason: str = Field(description="Why this action is recommended for this dataset")
    priority: Literal["high", "medium", "low"] = Field(description="Action priority")
    parameters: EngineeringActionParameters = Field(description="Action parameters")


class EngineeringRecommendationList(BaseModel):
    """List of feature engineering action recommendations from the AI agent."""
    reasoning: str = Field(
        description="Brief overall analysis of the dataset's feature engineering opportunities"
    )
    recommendations: List[EngineeringActionRecommendation] = Field(
        description="Ordered list of recommended feature engineering actions"
    )


# ── Mapping helpers ───────────────────────────────────────────

DOMAIN_MODEL_MAP = {
    "cleaning": CleaningRecommendationList,
    "conversion": ConversionRecommendationList,
    "feature_engineering": EngineeringRecommendationList,
}


def get_recommendation_model_for_domain(domain: str):
    """Get the correct Pydantic model for a domain."""
    model = DOMAIN_MODEL_MAP.get(domain)
    if not model:
        raise ValueError(f"Unknown domain: {domain}. Must be one of {list(DOMAIN_MODEL_MAP.keys())}")
    return model