"""
Deduplicates and merges static + AI agent recommendations.
If both systems recommend the same action, the priority is boosted
and the user sees a note that it was confirmed by both engines.
"""

from typing import List, Dict, Tuple


def deduplicate_recommendations(
    static_recs: List[Dict],
    ai_recs: List[Dict],
) -> List[Dict]:
    """
    Merge static and AI recommendations, deduplicating overlaps.

    When both engines recommend the same action (same action_type + same columns),
    we keep one merged entry with:
    - priority boosted to "high"
    - author set to "both"
    - description noting both engines agree

    Args:
        static_recs: Recommendations from static analyzers.
        ai_recs: Recommendations from AI agent.

    Returns:
        Deduplicated, merged list sorted by priority.
    """
    merged = []
    used_ai_ids = set()

    for static_rec in static_recs:
        match_found = False
        for ai_rec in ai_recs:
            if ai_rec.get("action_id") in used_ai_ids:
                continue

            if _is_same_action(static_rec, ai_rec):
                # Merge: boost priority, combine descriptions
                merged_rec = _merge_recommendations(static_rec, ai_rec)
                merged.append(merged_rec)
                used_ai_ids.add(ai_rec["action_id"])
                match_found = True
                break

        if not match_found:
            merged.append(static_rec)

    # Add remaining AI recs that didn't match any static rec
    for ai_rec in ai_recs:
        if ai_rec.get("action_id") not in used_ai_ids:
            merged.append(ai_rec)

    # Sort by priority
    priority_order = {"high": 0, "medium": 1, "low": 2}
    merged.sort(key=lambda r: priority_order.get(r.get("priority", "low"), 3))

    return merged


def _is_same_action(rec_a: Dict, rec_b: Dict) -> bool:
    """
    Determine if two recommendations target the same thing.
    Matches on action_type + target columns.
    """
    if rec_a.get("action_type") != rec_b.get("action_type"):
        return False

    params_a = rec_a.get("parameters", {})
    params_b = rec_b.get("parameters", {})

    # Compare target columns
    cols_a = _extract_columns(params_a)
    cols_b = _extract_columns(params_b)

    if cols_a and cols_b:
        # If they share at least 50% of columns, consider it the same action
        overlap = set(cols_a) & set(cols_b)
        total = set(cols_a) | set(cols_b)
        if len(total) > 0 and len(overlap) / len(total) >= 0.5:
            return True

    # If no columns to compare, match on action_type + strategy/method
    strategy_a = params_a.get("strategy") or params_a.get("method") or params_a.get("operation")
    strategy_b = params_b.get("strategy") or params_b.get("method") or params_b.get("operation")

    if strategy_a and strategy_b and strategy_a == strategy_b:
        return True

    return False


def _extract_columns(params: Dict) -> List[str]:
    """Extract column names from parameters."""
    cols = params.get("columns", [])
    if cols:
        return cols if isinstance(cols, list) else [cols]

    col = params.get("column")
    if col:
        return [col]

    subset = params.get("subset")
    if subset:
        return subset if isinstance(subset, list) else [subset]

    conversions = params.get("conversions")
    if conversions and isinstance(conversions, dict):
        return list(conversions.keys())

    return []


def _merge_recommendations(static_rec: Dict, ai_rec: Dict) -> Dict:
    """Merge two matching recommendations into one boosted entry."""
    # Use AI description as it's usually more detailed
    description = ai_rec.get("description", static_rec.get("description", ""))

    # Combine reasons
    static_reason = static_rec.get("reason", "")
    ai_reason = ai_rec.get("reason", "")
    combined_reason = (
        f"⚡ Both static analysis and AI agent recommend this action.\n\n"
        f"**Static analysis:** {static_reason}\n\n"
        f"**AI agent:** {ai_reason}"
    )

    # Always boost to high priority
    merged = {
        **ai_rec,  # Start with AI rec as base
        "action_id": static_rec.get("action_id", ai_rec.get("action_id")),
        "description": f"⚡ {description}",
        "author": "both",
        "priority": "high",
        "reason": combined_reason,
        "confirmed_by_static": True,
        "confirmed_by_ai": True,
    }

    return merged