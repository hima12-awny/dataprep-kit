"""
Pipeline JSON import/export with schema validation.
"""

import json
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timezone

from core.pipeline import Pipeline
from config.settings import settings


class PipelineIO:
    """Handles pipeline serialization and deserialization with validation."""

    REQUIRED_TOP_LEVEL_KEYS = ["schema_version", "pipeline_id", "steps"]
    REQUIRED_STEP_KEYS = ["action_type", "parameters"]

    @classmethod
    def export_pipeline(cls, pipeline: Pipeline, indent: int = 2) -> str:
        return pipeline.to_json(indent=indent)

    @classmethod
    def export_pipeline_bytes(cls, pipeline: Pipeline) -> bytes:
        return cls.export_pipeline(pipeline).encode("utf-8")

    @classmethod
    def import_pipeline(cls, json_content: str) -> Tuple[Optional[Pipeline], List[str]]:
        errors = []

        try:
            data = json.loads(json_content)
        except json.JSONDecodeError as e:
            return None, [f"Invalid JSON: {e}"]

        validation_errors = cls.validate_pipeline_json(data)
        if validation_errors:
            return None, validation_errors

        try:
            pipeline = Pipeline.from_dict(data)
            return pipeline, []
        except Exception as e:
            return None, [f"Failed to reconstruct pipeline: {e}"]

    @classmethod
    def import_pipeline_from_bytes(cls, content: bytes) -> Tuple[Optional[Pipeline], List[str]]:
        try:
            json_str = content.decode("utf-8")
        except UnicodeDecodeError:
            return None, ["File is not valid UTF-8 encoded text"]
        return cls.import_pipeline(json_str)

    @classmethod
    def validate_pipeline_json(cls, data: Dict) -> List[str]:
        errors = []

        if not isinstance(data, dict):
            return ["Pipeline JSON must be a dictionary"]

        for key in cls.REQUIRED_TOP_LEVEL_KEYS:
            if key not in data:
                errors.append(f"Missing required top-level key: '{key}'")

        version = data.get("schema_version")
        if version and version != settings.PIPELINE_SCHEMA_VERSION:
            errors.append(
                f"Schema version mismatch: file has '{version}', "
                f"app expects '{settings.PIPELINE_SCHEMA_VERSION}'"
            )

        steps = data.get("steps", [])
        if not isinstance(steps, list):
            errors.append("'steps' must be a list")
            return errors

        for i, step in enumerate(steps):
            step_errors = cls._validate_step(step, i)
            errors.extend(step_errors)

        return errors

    @classmethod
    def _validate_step(cls, step: Dict, index: int) -> List[str]:
        errors = []

        if not isinstance(step, dict):
            return [f"Step {index}: must be a dictionary"]

        for key in cls.REQUIRED_STEP_KEYS:
            if key not in step:
                errors.append(f"Step {index}: missing required key '{key}'")

        action_type = step.get("action_type")
        if action_type:
            from config.registry import ActionRegistry
            if not ActionRegistry.get(action_type):
                errors.append(
                    f"Step {index}: unknown action_type '{action_type}'"
                )

        author = step.get("author", "user")
        valid_authors = ["user", "ai", "ai_static", "ai_agent"]
        if author not in valid_authors:
            errors.append(
                f"Step {index}: invalid author '{author}'. Must be one of {valid_authors}"
            )

        params = step.get("parameters")
        if params is not None and not isinstance(params, dict):
            errors.append(f"Step {index}: 'parameters' must be a dictionary")

        return errors

    @classmethod
    def merge_pipelines(cls, *pipelines: Pipeline) -> Pipeline:
        merged = Pipeline()
        for pipeline in pipelines:
            for step in pipeline.steps:
                merged.add_step(step)
        return merged
