"""
Dataset wrapper that holds the dataframe, metadata, and snapshot history for undo/redo.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Any
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone

from config.settings import settings
from utils.id_generator import generate_snapshot_id
from utils.stats_helpers import StatsHelper


@dataclass
class Snapshot:
    """A frozen state of the dataset at a point in time."""
    snapshot_id: str
    dataframe: pd.DataFrame
    action_id: Optional[str]
    description: str
    timestamp: str

    def memory_usage_mb(self) -> float:
        return round(self.dataframe.memory_usage(deep=True).sum() / (1024 * 1024), 2)


class Dataset:
    """
    Wrapper around a pandas DataFrame with:
    - Metadata tracking (source file, import config)
    - Snapshot-based undo/redo
    - Quick profiling access
    """

    def __init__(
        self,
        df: pd.DataFrame,
        source_name: str = "unknown",
        import_config: Optional[Dict] = None,
    ):
        self._df: pd.DataFrame = df.copy()
        self.source_name: str = source_name
        self.import_config: Dict = import_config or {}
        self.created_at: str = datetime.now(timezone.utc).isoformat()

        # Undo/Redo stacks
        self._undo_stack: List[Snapshot] = []
        self._redo_stack: List[Snapshot] = []

        # Save initial snapshot
        self._save_snapshot(action_id=None, description="Initial import")

    # ── Properties ────────────────────────────────────────────

    @property
    def df(self) -> pd.DataFrame:
        """Access the current dataframe."""
        return self._df

    @df.setter
    def df(self, new_df: pd.DataFrame):
        """Set a new dataframe (use update() instead for undo support)."""
        self._df = new_df

    @property
    def shape(self) -> tuple:
        return self._df.shape

    @property
    def columns(self) -> List[str]:
        return self._df.columns.tolist()

    @property
    def dtypes(self) -> Dict[str, str]:
        return {col: str(dtype) for col, dtype in self._df.dtypes.items()}

    @property
    def numeric_columns(self) -> List[str]:
        return self._df.select_dtypes(include=np.number).columns.tolist()

    @property
    def categorical_columns(self) -> List[str]:
        return self._df.select_dtypes(include=["object", "category"]).columns.tolist()

    @property
    def datetime_columns(self) -> List[str]:
        return self._df.select_dtypes(include=["datetime64"]).columns.tolist()

    # ── Core Operations ───────────────────────────────────────

    def update(self, new_df: pd.DataFrame, action_id: str, description: str):
        """
        Update the dataset with a new dataframe, saving the previous state for undo.
        This is the ONLY way actions should modify the dataset.
        """
        self._save_snapshot(action_id=action_id, description=description)
        self._df = new_df.copy()
        self._redo_stack.clear()  # New action invalidates redo history

    def undo(self) -> bool:
        """
        Revert to the previous state.
        Returns True if undo was successful.
        """
        if len(self._undo_stack) <= 1:
            return False  # Can't undo past initial import

        # Save current state to redo stack
        current_snapshot = Snapshot(
            snapshot_id=generate_snapshot_id(),
            dataframe=self._df.copy(),
            action_id="redo_point",
            description="Before undo",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._redo_stack.append(current_snapshot)

        # Pop the last snapshot and restore it
        self._undo_stack.pop()  # Remove the snapshot that represents current state
        previous = self._undo_stack[-1]
        self._df = previous.dataframe.copy()
        return True

    def redo(self) -> bool:
        """
        Re-apply the last undone action.
        Returns True if redo was successful.
        """
        if not self._redo_stack:
            return False

        snapshot = self._redo_stack.pop()
        self._save_snapshot(action_id="redo", description="Redo")
        self._df = snapshot.dataframe.copy()
        return True

    @property
    def can_undo(self) -> bool:
        return len(self._undo_stack) > 1

    @property
    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    @property
    def undo_steps_count(self) -> int:
        return max(0, len(self._undo_stack) - 1)

    # ── Profiling ─────────────────────────────────────────────

    def overview(self) -> Dict:
        """Get dataset overview statistics."""
        return StatsHelper.dataframe_overview(self._df)

    def column_stats(self, column: str) -> Dict:
        """Get detailed stats for a single column."""
        if column not in self._df.columns:
            return {}
        return StatsHelper.get_column_stats(df=self._df, col_name=column)

    def all_column_stats(self) -> Dict[str, Dict]:
        """Get stats for all columns."""
        return {col: self.column_stats(col) for col in self._df.columns}

    def head(self, n: int = 10) -> pd.DataFrame:
        return self._df.head(n)

    def sample(self, n: int = 10) -> pd.DataFrame:
        actual_n = min(n, len(self._df))
        return self._df.sample(n=actual_n, random_state=42) if actual_n > 0 else self._df

    # ── Serialization ─────────────────────────────────────────

    def get_metadata(self) -> Dict:
        """Return metadata for pipeline export."""
        return {
            "source_name": self.source_name,
            "import_config": self.import_config,
            "created_at": self.created_at,
            "original_shape": {
                "rows": self._undo_stack[0].dataframe.shape[0] if self._undo_stack else self.shape[0],
                "columns": self._undo_stack[0].dataframe.shape[1] if self._undo_stack else self.shape[1],
            },
            "current_shape": {
                "rows": self.shape[0],
                "columns": self.shape[1],
            },
        }

    # ── Private ───────────────────────────────────────────────

    def _save_snapshot(self, action_id: Optional[str], description: str):
        """Save a snapshot of the current state."""
        snapshot = Snapshot(
            snapshot_id=generate_snapshot_id(),
            dataframe=self._df.copy(),
            action_id=action_id,
            description=description,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._undo_stack.append(snapshot)

        # Trim undo stack if too large
        if len(self._undo_stack) > settings.MAX_UNDO_STEPS:
            self._undo_stack = self._undo_stack[-settings.MAX_UNDO_STEPS:]
