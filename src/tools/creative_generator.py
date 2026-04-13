"""Creative generator tools — strategy loading and content direction."""

from __future__ import annotations

from langchain_core.tools import tool


@tool
def load_strategy(strategy_info: str) -> str:
    """加载并应用一条推荐策略。

    Args:
        strategy_info: 策略名称或描述

    Returns:
        策略的详细执行步骤
    """
    return f"策略 '{strategy_info}' 已加载。请按照策略中的执行步骤进行推荐。"
