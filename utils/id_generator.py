"""
Short unique ID generation for action_ids.
"""

import uuid
import hashlib
from datetime import datetime, timezone


def generate_action_id(prefix: str = "") -> str:
    """
    Generate a short, unique action ID.
    Format: optional prefix + 8-char hex string.
    """
    raw = uuid.uuid4().hex[:8]
    if prefix:
        return f"{prefix}_{raw}"
    return raw


def generate_pipeline_id() -> str:
    """Generate a unique pipeline ID with timestamp component."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    short_uuid = uuid.uuid4().hex[:6]
    return f"pipeline_{ts}_{short_uuid}"


def generate_snapshot_id() -> str:
    """Generate a unique snapshot ID."""
    return f"snap_{uuid.uuid4().hex[:10]}"