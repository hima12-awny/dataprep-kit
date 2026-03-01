"""
Display formatting helpers for the Streamlit UI.
"""

from typing import Any, Optional


def format_number(value: float, decimals: int = 2) -> str:
    """Format a number with commas and decimal places."""
    if value is None:
        return "N/A"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:,.{decimals}f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:,.{decimals}f}K"
    return f"{value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format a decimal or percentage value."""
    if value is None:
        return "N/A"
    if abs(value) <= 1.0:
        return f"{value * 100:.{decimals}f}%"
    return f"{value:.{decimals}f}%"


def truncate_string(text: str, max_length: int = 50) -> str:
    """Truncate a string and add ellipsis if needed."""
    if not text or len(text) <= max_length:
        return text or ""
    return text[: max_length - 3] + "..."


def format_bytes(size_bytes: int) -> str:
    """Format byte size to human readable."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.1f} MB"
    else:
        return f"{size_bytes / (1024 ** 3):.1f} GB"


def author_badge_html(author: str) -> str:
    """Return HTML badge for action author."""
    badge_map = {
        "user": ("User", "badge-user"),
        "ai_static": ("AI Static", "badge-ai-static"),
        "ai_agent": ("AI Agent", "badge-ai-agent"),
        "ai": ("AI", "badge-ai-static"),
        "both": ("AI + User", "badge-ai-user"),
    }
    label, css_class = badge_map.get(author, ("Unknown", "badge-user"))
    return f'<span class="author-badge {css_class}">{label}</span>'


def priority_css_class(priority: str) -> str:
    """Return CSS class for recommendation priority."""
    return {
        "high": "priority-high",
        "medium": "priority-medium",
        "low": "priority-low",
    }.get(priority, "priority-low")
