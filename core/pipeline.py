"""
Pipeline: an ordered list of actions that can be executed, exported, and replayed.
"""

import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from copy import deepcopy

from config.settings import settings
from utils.id_generator import generate_pipeline_id, generate_action_id

if TYPE_CHECKING:
    from core.dataset import Dataset
    from actions.base import BaseAction 


class PipelineStep:
    """A single step in the pipeline."""

    def __init__(
        self,
        action_id: str,
        action_type: str,
        description: str,
        author: str,
        parameters: Dict[str, Any],
        timestamp: Optional[str] = None,
        preview_only: bool = False,
        enabled: bool = True,
    ):
        self.action_id = action_id
        self.action_type = action_type
        self.description = description
        self.author = author
        self.parameters = parameters
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()
        self.preview_only = preview_only
        self.enabled = enabled
        self.execution_result: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Serialize to the standard action JSON schema."""
        return {
            "action_id": self.action_id,
            "action_type": self.action_type,
            "description": self.description,
            "author": self.author,
            "timestamp": self.timestamp,
            "parameters": self.parameters,
            "preview_only": self.preview_only,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "PipelineStep":
        """Deserialize from JSON dict."""
        return cls(
            action_id=data.get("action_id", generate_action_id()),
            action_type=data["action_type"],
            description=data.get("description", ""),
            author=data.get("author", "user"),
            parameters=data.get("parameters", {}),
            timestamp=data.get("timestamp"),
            preview_only=data.get("preview_only", False),
            enabled=data.get("enabled", True),
        )


class Pipeline:
    """
    Ordered collection of PipelineSteps that can be:
    - Built interactively (step by step)
    - Exported as JSON
    - Imported from JSON and replayed on new data
    """

    def __init__(self, pipeline_id: Optional[str] = None):
        self.pipeline_id: str = pipeline_id or generate_pipeline_id()
        self.steps: List[PipelineStep] = []
        self.created_at: str = datetime.now(timezone.utc).isoformat()
        self.metadata: Dict = {}
        self.change_log: List[Dict] = []

    # ── Step Management ───────────────────────────────────────

    def add_step(self, step: PipelineStep) -> None:
        """Add a step to the end of the pipeline."""
        self.steps.append(step)
        self._log_change("add", step.action_id, step.description)

    def add_action(
        self,
        action_type: str,
        description: str,
        parameters: Dict,
        author: str = "user",
        preview_only: bool = False,
    ) -> PipelineStep:
        """Create and add a new step from raw parameters."""
        step = PipelineStep(
            action_id=generate_action_id(),
            action_type=action_type,
            description=description,
            author=author,
            parameters=parameters,
            preview_only=preview_only,
        )
        self.add_step(step)
        return step

    def remove_step(self, action_id: str) -> bool:
        """Remove a step by its action_id."""
        for i, step in enumerate(self.steps):
            if step.action_id == action_id:
                removed = self.steps.pop(i)
                self._log_change("remove", action_id, removed.description)
                return True
        return False

    def reorder_step(self, action_id: str, new_index: int) -> bool:
        """Move a step to a new position."""
        for i, step in enumerate(self.steps):
            if step.action_id == action_id:
                self.steps.pop(i)
                self.steps.insert(min(new_index, len(self.steps)), step)
                self._log_change("reorder", action_id, f"Moved to position {new_index}")
                return True
        return False

    def toggle_step(self, action_id: str) -> bool:
        """Enable/disable a step."""
        for step in self.steps:
            if step.action_id == action_id:
                step.enabled = not step.enabled
                status = "enabled" if step.enabled else "disabled"
                self._log_change("toggle", action_id, f"Step {status}")
                return True
        return False

    def get_step(self, action_id: str) -> Optional[PipelineStep]:
        """Get a step by its action_id."""
        for step in self.steps:
            if step.action_id == action_id:
                return step
        return None

    @property
    def enabled_steps(self) -> List[PipelineStep]:
        """Return only enabled steps."""
        return [s for s in self.steps if s.enabled]

    @property
    def step_count(self) -> int:
        return len(self.steps)

    # ── Execution ─────────────────────────────────────────────

    def execute(self, dataset: "Dataset") -> Dict:
        """
        Execute all enabled steps on the dataset sequentially.
        Returns execution report.
        """
        from config.registry import ActionRegistry

        report = {
            "pipeline_id": self.pipeline_id,
            "total_steps": len(self.enabled_steps),
            "executed": 0,
            "skipped": 0,
            "failed": 0,
            "errors": [],
        }

        for step in self.enabled_steps:
            action_class = ActionRegistry.get(step.action_type)
            if not action_class:
                report["failed"] += 1
                report["errors"].append({
                    "action_id": step.action_id,
                    "error": f"Unknown action_type: {step.action_type}",
                })
                continue

            try:
                action = action_class()
                validation_errors = action.validate(dataset.df, step.parameters)

                if validation_errors:
                    report["failed"] += 1
                    report["errors"].append({
                        "action_id": step.action_id,
                        "error": f"Validation failed: {validation_errors}",
                    })
                    continue

                result_df = action.execute(dataset.df, step.parameters)
                dataset.update(result_df, step.action_id, step.description)

                step.execution_result = {
                    "status": "success",
                    "rows_before": dataset.shape[0],
                    "rows_after": result_df.shape[0],
                }
                report["executed"] += 1

            except Exception as e:
                report["failed"] += 1
                report["errors"].append({
                    "action_id": step.action_id,
                    "error": str(e),
                })

        return report

    def execute_single_step(
        self, step: PipelineStep, dataset: "Dataset"
    ) -> Dict:
        """Execute a single step. Used for interactive approve-and-run."""
        from config.registry import ActionRegistry

        action_class = ActionRegistry.get(step.action_type)
        if not action_class:
            return {"status": "error", "error": f"Unknown action_type: {step.action_type}"}

        try:
            action = action_class()
            errors = action.validate(dataset.df, step.parameters)
            if errors:
                return {"status": "error", "error": f"Validation: {errors}"}

            result_df = action.execute(dataset.df, step.parameters)
            rows_before = len(dataset.df)
            dataset.update(result_df, step.action_id, step.description)

            step.execution_result = {"status": "success"}
            return {
                "status": "success",
                "rows_before": rows_before,
                "rows_after": len(result_df),
                "cols_before": dataset.shape[1],
                "cols_after": result_df.shape[1],
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def preview_step(self, step: PipelineStep, dataset: "Dataset") -> Dict:
        """Preview what a step would do without mutating the dataset."""
        from config.registry import ActionRegistry

        action_class = ActionRegistry.get(step.action_type)
        if not action_class:
            return {"status": "error", "error": f"Unknown action_type: {step.action_type}"}

        try:
            action = action_class()
            errors = action.validate(dataset.df, step.parameters)
            if errors:
                return {"status": "error", "error": f"Validation: {errors}"}

            preview_result = action.preview(dataset.df, step.parameters)
            return {"status": "success", **preview_result}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # ── Serialization (Export/Import) ─────────────────────────

    def to_dict(self) -> Dict:
        """Serialize the entire pipeline to a dict for JSON export."""
        return {
            "schema_version": settings.PIPELINE_SCHEMA_VERSION,
            "pipeline_id": self.pipeline_id,
            "created_at": self.created_at,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "metadata": self.metadata,
            "steps": [step.to_dict() for step in self.steps],
            "change_log": self.change_log,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: Dict) -> "Pipeline":
        """Reconstruct a pipeline from a dict."""
        pipeline = cls(pipeline_id=data.get("pipeline_id"))
        pipeline.created_at = data.get("created_at", pipeline.created_at)
        pipeline.metadata = data.get("metadata", {})
        pipeline.change_log = data.get("change_log", [])

        for step_data in data.get("steps", []):
            step = PipelineStep.from_dict(step_data)
            pipeline.steps.append(step)

        return pipeline

    @classmethod
    def from_json(cls, json_string: str) -> "Pipeline":
        """Reconstruct a pipeline from a JSON string."""
        data = json.loads(json_string)
        return cls.from_dict(data)

    # ── Change Log ────────────────────────────────────────────

    def _log_change(self, change_type: str, action_id: str, description: str):
        """Record a change in the audit trail."""
        self.change_log.append({
            "change_type": change_type,
            "action_id": action_id,
            "description": description,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def get_change_log(self) -> List[Dict]:
        return list(self.change_log)

    # ── Utilities ─────────────────────────────────────────────

    def clear(self):
        """Remove all steps."""
        self.steps.clear()
        self._log_change("clear", "all", "Pipeline cleared")

    def duplicate(self) -> "Pipeline":
        """Create a copy of this pipeline with a new ID."""
        new_pipeline = Pipeline()
        new_pipeline.metadata = deepcopy(self.metadata)
        for step in self.steps:
            new_step = PipelineStep.from_dict(step.to_dict())
            new_step.action_id = generate_action_id()
            new_pipeline.steps.append(new_step)
        return new_pipeline