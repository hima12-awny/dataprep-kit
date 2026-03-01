# 🛠️ DataPrep Kit - Developer Guide

This guide is intended for engineers and contributors who want to understand the internal architecture of **DataPrep Kit** or extend its functionality with new actions and AI agents.

## 🏗️ Core Architecture

DataPrep Kit is built on a modular, event-driven architecture that separates specialized domains (Cleaning, Conversion, Engineering) from the core execution engine.

### 1. State Management (`core/state.py` & `core/dataset.py`)
The application uses a **Snapshot-based State Pattern**.
- **`Dataset`**: A wrapper around a pandas DataFrame that maintains an `_undo_stack` and `_redo_stack`.
- **`Snapshot`**: Every destructive or transformative action creates a `Snapshot`. This includes the DataFrame state, a unique `action_id`, and a descriptive timestamp.
- **`StateManager`**: A singleton that provides typed access to the Streamlit `session_state`, ensuring the UI and backend remain in sync.

### 2. The Pipeline Engine (`core/pipeline.py`)
The Pipeline is a serializable sequence of `PipelineStep` objects.
- **Immutability**: Once an action is added, it is assigned a UUID.
- **Replayability**: Pipelines are exported as JSON. When imported, the engine uses the `ActionRegistry` to map `action_type` strings back to their Python classes and executes them sequentially on the raw data.
- **Preview Mode**: Every action must implement a `preview()` method, allowing users to see a "diff" (e.g., rows dropped, columns changed) before committing to the state.

### 3. Action Registry (`config/registry.py`)
We use a decorator-based registration system:
```python
@register_action("your_action_name")
class YourAction(BaseAction):
    ...
```
This allows the engine to dynamically discover and instantiate actions without hardcoded maps.

---

## 🚀 Extending functionality

### Adding a New Action
1.  **Subclass `BaseAction`**: Located in `actions/base.py`.
2.  **Define Domain**: Choose `cleaning`, `conversion`, or `feature_engineering`.
3.  **Implement `execute()`**: Perform the pandas transformation.
4.  **Implement `preview()`**: Return a summary of changes (e.g., `{'rows_affected': 10}`).
5.  **Register**: Use the `@register_action` decorator.

Example:
```python
@register_action("custom_filter")
class CustomFilterAction(BaseAction):
    domain = "cleaning"
    
    def execute(self, df, params):
        # Your logic here
        return df.query(params['query'])
```

### AI Agent Integration (`recommendations/ai_agent/`)
The AI layer uses **`pydantic-ai`** for structured, type-safe LLM interactions.
- **`BaseAgent`**: Handles API key injection into env vars and model string formatting.
- **`ContextBuilder`**: Compresses dataset statistics, schema info, and sample rows into a high-density prompt for the LLM.
- **`models.py`**: Defines Pydantic schemas for LLM outputs, ensuring the UI always receives valid JSON.

---

## 🧪 Testing

We use `pytest` for all backend logic.
- **Unit Tests**: Test individual actions in `tests/test_actions/`.
- **Integration Tests**: Test the full pipeline flow in `tests/test_core/`.

Run tests with:
```bash
pytest tests/
```

## 🛠️ Performance Considerations
- **Memory**: Snapshots store copies of DataFrames. For very large datasets, the `MAX_UNDO_STEPS` (default 30) should be tuned.
- **Sampling**: AI agents and type detectors use deterministic sampling (default 1000 rows) to ensure low latency even on million-row files.
