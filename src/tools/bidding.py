"""Bidding tools — bid priority strategy for ad delivery."""

from __future__ import annotations

import json

from langchain_core.tools import tool


_BID_MODES = {"conservative", "balanced", "aggressive"}


@tool
def set_bid_strategy(
    audience_size: int,
    competition_level: str,
    mode: str = "balanced",
) -> str:
    """根据受众规模和竞争强度设定出价策略。

    Args:
        audience_size: 目标受众规模（人数）
        competition_level: 竞争强度 ("low" | "medium" | "high")
        mode: 出价模式 ("conservative" | "balanced" | "aggressive")

    Returns:
        出价策略建议（含建议分数和理由）
    """
    if mode not in _BID_MODES:
        mode = "balanced"

    base = {"conservative": 0.6, "balanced": 1.0, "aggressive": 1.5}[mode]
    comp_mult = {"low": 0.8, "medium": 1.0, "high": 1.3}.get(competition_level, 1.0)

    size_factor = 1.0
    if audience_size < 1000:
        size_factor = 1.2  # small audience → higher bid for reach
    elif audience_size > 100000:
        size_factor = 0.85  # large audience → spread budget

    bid_score = round(base * comp_mult * size_factor, 2)

    return json.dumps(
        {
            "bid_score": bid_score,
            "mode": mode,
            "rationale": (
                f"受众 {audience_size} 人，{competition_level} 竞争，{mode} 模式 → "
                f"建议出价优先级 {bid_score}"
            ),
        },
        ensure_ascii=False,
    )
