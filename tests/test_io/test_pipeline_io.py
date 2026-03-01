"""
Tests for PipelineIO: export, import, validation.
"""

import pytest
import json

from data_io.pipeline_io import PipelineIO
from core.pipeline import Pipeline


@pytest.fixture
def sample_pipeline():
    pipeline = Pipeline()
    pipeline.add_action(
        action_type="handle_missing",
        description="Impute age",
        parameters={"columns": ["age"], "strategy": "median"},
        author="user",
    )
    pipeline.add_action(
        action_type="handle_duplicates",
        description="Remove dups",
        parameters={"keep": "first"},
        author="ai_static",
    )
    return pipeline


class TestPipelineIO:
    def test_export_returns_string(self, sample_pipeline):
        result = PipelineIO.export_pipeline(sample_pipeline)
        assert isinstance(result, str)
        data = json.loads(result)
        assert "steps" in data

    def test_export_bytes(self, sample_pipeline):
        result = PipelineIO.export_pipeline_bytes(sample_pipeline)
        assert isinstance(result, bytes)

    def test_import_valid(self, sample_pipeline):
        json_str = PipelineIO.export_pipeline(sample_pipeline)
        pipeline, errors = PipelineIO.import_pipeline(json_str)
        assert len(errors) == 0
        assert pipeline is not None
        assert pipeline.step_count == 2

    def test_import_from_bytes(self, sample_pipeline):
        bytes_content = PipelineIO.export_pipeline_bytes(sample_pipeline)
        pipeline, errors = PipelineIO.import_pipeline_from_bytes(bytes_content)
        assert len(errors) == 0
        assert pipeline is not None

    def test_import_invalid_json(self):
        pipeline, errors = PipelineIO.import_pipeline("not valid json{")
        assert pipeline is None
        assert len(errors) > 0

    def test_import_missing_keys(self):
        json_str = json.dumps({"some_key": "value"})
        pipeline, errors = PipelineIO.import_pipeline(json_str)
        assert len(errors) > 0

    def test_roundtrip_preserves_data(self, sample_pipeline):
        json_str = PipelineIO.export_pipeline(sample_pipeline)
        restored, errors = PipelineIO.import_pipeline(json_str)

        assert len(errors) == 0
        assert restored.pipeline_id == sample_pipeline.pipeline_id
        assert restored.step_count == sample_pipeline.step_count

        for orig, rest in zip(sample_pipeline.steps, restored.steps):
            assert orig.action_type == rest.action_type
            assert orig.parameters == rest.parameters
            assert orig.author == rest.author
            assert orig.description == rest.description

    def test_validate_valid_pipeline(self, sample_pipeline):
        data = json.loads(PipelineIO.export_pipeline(sample_pipeline))
        errors = PipelineIO.validate_pipeline_json(data)
        assert len(errors) == 0

    def test_validate_invalid_step(self):
        data = {
            "schema_version": "1.0.0",
            "pipeline_id": "test",
            "steps": [
                {"not_action_type": "bad"}
            ],
        }
        errors = PipelineIO.validate_pipeline_json(data)
        assert len(errors) > 0

    def test_merge_pipelines(self):
        p1 = Pipeline()
        p1.add_action("handle_missing", "Step 1", {"strategy": "mean"})

        p2 = Pipeline()
        p2.add_action("handle_duplicates", "Step 2", {"keep": "first"})

        merged = PipelineIO.merge_pipelines(p1, p2)
        assert merged.step_count == 2
