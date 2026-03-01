"""
Parses LLM output into validated Action JSON dicts — Phase 2 placeholder.
"""

import json
from typing import Dict, List, Optional

from utils.id_generator import generate_action_id


class ResponseParser:
    """
    Parses raw LLM text output into structured Action dicts.
    Phase 2 will handle:
    - JSON extraction from markdown code blocks
    - Schema validation against toolset schemas
    - Fallback parsing for malformed responses
    """

    @staticmethod
    def parse(raw_response: str) -> List[Dict]:
        """Parse LLM response into action dicts."""
        # Phase 2 implementation
        raise NotImplementedError("Phase 2")

    @staticmethod
    def _extract_json_blocks(text: str) -> List[str]:  # type: ignore
        """Extract JSON blocks from markdown-formatted text."""
        # Phase 2
        pass

    @staticmethod
    def _validate_action(action: Dict) -> Optional[Dict]:
        """Validate a single action dict against the schema."""
        # Phase 2
        pass
