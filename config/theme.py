"""
UI theme settings for consistent styling across all pages.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    """UI theme constants."""

    # ── Colors ────────────────────────────────────────────────
    PRIMARY: str = "#4F46E5"       # Indigo
    SECONDARY: str = "#7C3AED"     # Violet
    SUCCESS: str = "#10B981"       # Green
    WARNING: str = "#F59E0B"       # Amber
    DANGER: str = "#EF4444"        # Red
    INFO: str = "#3B82F6"          # Blue
    MUTED: str = "#6B7280"         # Gray

    # ── Author Badge Colors ───────────────────────────────────
    AUTHOR_USER: str = "#3B82F6"
    AUTHOR_AI_STATIC: str = "#8B5CF6"
    AUTHOR_AI_AGENT: str = "#EC4899"

    # ── Recommendation Priority ───────────────────────────────
    PRIORITY_HIGH: str = "#EF4444"
    PRIORITY_MEDIUM: str = "#F59E0B"
    PRIORITY_LOW: str = "#10B981"

    # ── Layout ────────────────────────────────────────────────
    SIDEBAR_WIDTH: int = 300
    PREVIEW_TABLE_HEIGHT: int = 400
    MAX_COLUMNS_DISPLAY: int = 50

    # ── Custom CSS ────────────────────────────────────────────
    @staticmethod
    def get_custom_css() -> str:
        return """
        <style>
            .stApp > header {
                background-color: transparent;
            }
            .action-card {
                border: 1px solid #e5e7eb;
                border-radius: 12px;
                padding: 1rem;
                margin-bottom: 0.75rem;
                background: #ffffff;
                box-shadow: 0 1px 3px rgba(0,0,0,0.06);
                transition: box-shadow 0.2s;
            }
            .action-card:hover {
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }
            .author-badge {
                display: inline-block;
                padding: 2px 10px;
                border-radius: 12px;
                font-size: 0.75rem;
                font-weight: 600;
                color: white;
            }
            .badge-user { background-color: #3B82F6; }
            .badge-ai-static { background-color: #8B5CF6; }
            .badge-ai-agent { background-color: #EC4899; }
            .priority-high { border-left: 4px solid #EF4444; }
            .priority-medium { border-left: 4px solid #F59E0B; }
            .priority-low { border-left: 4px solid #10B981; }
            .metric-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 1rem;
                border-radius: 10px;
                color: white;
                text-align: center;
            }
            .step-indicator {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                padding: 0.5rem;
                border-radius: 8px;
                margin-bottom: 0.25rem;
            }
            .step-indicator.active {
                background-color: #EEF2FF;
                border: 1px solid #4F46E5;
            }
            div[data-testid="stSidebarContent"] {
                padding-top: 1rem;
            }
        </style>
        """


theme = Theme()