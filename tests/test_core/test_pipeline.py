"""
Tests for the Pipeline class: adding steps, execution, export, import, replay.
"""

import json
import pytest
import pandas as pd

from core.pipeline import Pipeline, PipelineStep
from core.dataset import Dataset


class TestPipelineStep:
    def test_to_dict(self):
        step = PipelineStep(
            action_id="test123",
            action_type="handle_missing",
            description="Test step",
            author="user",
            parameters={"columns": ["age"], "strategy": "mean"},
        )
        d = step.to_dict()
        assert d["action_id"] == "test123"
        assert d["action_type"] == "handle_missing"
        assert d["author"] == "user"
        assert d["parameters"]["strategy"] == "mean"

    def test_from_dict(self):
        data = {
            "action_id": "abc",
            "action_type": "handle_duplicates",
            "description": "Remove dups",
            "author": "ai_static",
            "parameters": {"keep": "first"},
        }
        step = PipelineStep.from_dict(data)
        assert step.action_type == "handle_duplicates"
        assert step.author == "ai_static"
        assert step.enabled is True

    def test_roundtrip(self):
        step = PipelineStep(
            action_id="rt1",
            action_type="type_casting",
            description="Cast age to int",
            author="user",
            parameters={"conversions": {"age": "int64"}},
        )
        d = step.to_dict()
        restored = PipelineStep.from_dict(d)
        assert restored.action_id == step.action_id
        assert restored.parameters == step.parameters


class TestPipeline:
    def test_add_step(self, empty_pipeline):
        empty_pipeline.add_action(
            action_type="handle_missing",
            description="test",
            parameters={"strategy": "mean"},
        )
        assert empty_pipeline.step_count == 1

    def test_remove_step(self, sample_pipeline):
        action_id = sample_pipeline.steps[0].action_id
        result = sample_pipeline.remove_step(action_id)
        assert result is True
        assert sample_pipeline.step_count == 1

    def test_toggle_step(self, sample_pipeline):
        action_id = sample_pipeline.steps[0].action_id
        sample_pipeline.toggle_step(action_id)
        assert sample_pipeline.steps[0].enabled is False
        assert len(sample_pipeline.enabled_steps) == 1

    def test_reorder_step(self, sample_pipeline):
        first_id = sample_pipeline.steps[0].action_id
        sample_pipeline.reorder_step(first_id, 1)
        assert sample_pipeline.steps[1].action_id == first_id

    def test_export_json(self, sample_pipeline):
        json_str = sample_pipeline.to_json()
        data = json.loads(json_str)
        assert "schema_version" in data
        assert "pipeline_id" in data
        assert len(data["steps"]) == 2

    def test_import_json(self, sample_pipeline):
        json_str = sample_pipeline.to_json()
        restored = Pipeline.from_json(json_str)
        assert restored.pipeline_id == sample_pipeline.pipeline_id
        assert restored.step_count == sample_pipeline.step_count

    def test_roundtrip_json(self, sample_pipeline):
        json_str = sample_pipeline.to_json()
        restored = Pipeline.from_json(json_str)
        for orig, rest in zip(sample_pipeline.steps, restored.steps):
            assert orig.action_type == rest.action_type
            assert orig.parameters == rest.parameters
            assert orig.author == rest.author

    def test_change_log(self, empty_pipeline):
        empty_pipeline.add_action(
            action_type="handle_missing",
            description="test",
            parameters={},
        )
        log = empty_pipeline.get_change_log()
        assert len(log) == 1
        assert log[0]["change_type"] == "add"

    def test_clear(self, sample_pipeline):
        sample_pipeline.clear()
        assert sample_pipeline.step_count == 0

    def test_duplicate(self, sample_pipeline):
        dup = sample_pipeline.duplicate()
        assert dup.pipeline_id != sample_pipeline.pipeline_id
        assert dup.step_count == sample_pipeline.step_count
        # IDs should be different
        for orig, new in zip(sample_pipeline.steps, dup.steps):
            assert orig.action_id != new.action_id