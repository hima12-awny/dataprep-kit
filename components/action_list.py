"""
Renders a scrollable list of action recommendation cards.
Preview is shown directly below the card that triggered it.
"""

import streamlit as st
from typing import Dict, List, Callable, Optional

from components.action_card import render_action_card
from components.preview_table import render_preview_table


def render_action_list(
    recommendations: List[Dict],
    on_approve: Optional[Callable] = None,
    on_reject: Optional[Callable] = None,
    on_preview: Optional[Callable] = None,
    title: str = "Recommendations",
    key_prefix: str = "rec",
    show_filters: bool = True,
):
    """
    Render a list of recommendation cards with optional filtering.
    Preview renders inline directly below the card that triggered it.
    """
    if not recommendations:
        st.info("✨ No recommendations at this time. Your data looks good!")
        return

    # ── Filters ───────────────────────────────────────────────
    filtered = recommendations
    if show_filters and len(recommendations) > 3:
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            priority_filter = st.multiselect(
                "Priority",
                options=["high", "medium", "low"],
                default=["high", "medium", "low"],
                key=f"{key_prefix}_priority_filter",
            )

        with col2:
            action_types = list(set(r.get("action_type", "")
                                for r in recommendations))
            type_filter = st.multiselect(
                "Action Type",
                options=sorted(action_types),
                default=action_types,
                key=f"{key_prefix}_type_filter",
            )

        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            st.caption(f"{len(recommendations)} total")

        filtered = [
            r for r in recommendations
            if r.get("priority", "low") in priority_filter
            and r.get("action_type", "") in type_filter
        ]

    # ── Header ────────────────────────────────────────────────
    count_high = sum(1 for r in filtered if r.get("priority") == "high")
    count_med = sum(1 for r in filtered if r.get("priority") == "medium")
    count_low = sum(1 for r in filtered if r.get("priority") == "low")

    st.markdown(
        f"### {title} ({len(filtered)})\n"
        f"🔴 {count_high} high · 🟡 {count_med} medium · 🟢 {count_low} low"
    )

    # ── Track which card's preview is currently open ──────────
    active_preview_key = f"{key_prefix}_active_preview_id"

    # ── Cards + Inline Preview ────────────────────────────────
    for i, rec in enumerate(filtered):
        action_id = rec.get("action_id", f"unknown_{i}")
        card_key = f"{key_prefix}_{i}"
        preview_data_key = f"{key_prefix}_preview_{action_id}"

        # Render the card
        render_action_card(
            recommendation=rec,
            on_approve=on_approve,
            on_reject=on_reject,
            on_preview=lambda r, _aid=action_id, _pdk=preview_data_key, _apk=active_preview_key: (
                _handle_inline_preview(r, _aid, _pdk, _apk, on_preview)
            ),
            show_controls=True,
            key_prefix=card_key,
        )

        # ── Render preview DIRECTLY below this card if active ─
        active_id = st.session_state.get(active_preview_key)
        if active_id == action_id and preview_data_key in st.session_state:
            preview_data = st.session_state[preview_data_key]

            if preview_data is not None:
                with st.container(border=True):
                    # Compact header with close button
                    col_title, col_close = st.columns([5, 1])
                    with col_title:
                        desc = rec.get("description", "Action")
                        st.markdown(f"**👁️ Preview:** _{desc}_")
                    with col_close:
                        if st.button(
                            "✖️ Close",
                            key=f"{card_key}_close_preview",
                            width='stretch',
                        ):
                            st.session_state[preview_data_key] = None
                            st.session_state[active_preview_key] = None
                            st.rerun()

                    # Render the preview content
                    render_preview_table(preview_data, compact=True)


def _handle_inline_preview(
    recommendation: Dict,
    action_id: str,
    preview_data_key: str,
    active_preview_key: str,
    on_preview_callback: Optional[Callable],
):
    """
    Handle preview button click:
    - If same card is already previewed → toggle off
    - Otherwise → generate preview and show below this card
    """
    current_active = st.session_state.get(active_preview_key)

    # Toggle off if clicking same card's preview again
    if current_active == action_id:
        st.session_state[active_preview_key] = None
        st.session_state[preview_data_key] = None
        st.rerun()
        return

    # Close any previously active preview
    if current_active:
        # Clear old preview data
        for key in list(st.session_state.keys()):
            if key.startswith(active_preview_key.rsplit("_active_preview_id", 1)[0] + "_preview_"):
                st.session_state[key] = None

    # Generate new preview via the page callback
    if on_preview_callback:
        on_preview_callback(recommendation)

    # Mark this card as the active preview
    st.session_state[active_preview_key] = action_id

    st.rerun()
