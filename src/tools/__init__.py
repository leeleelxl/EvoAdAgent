"""Ad tools — specialized tool modules for the LangGraph executor.

Each tool is a LangChain ``@tool``-decorated function. ``get_all_tools()``
aggregates them for the executor's ToolNode.

If a ``UserProfileStore`` is passed to ``get_all_tools()``, an additional
``find_similar_users`` tool is added that queries the L2 FAISS index — this
is how the L2 memory layer actually gets consumed inside the ReAct loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.tools.audience_analyzer import analyze_audience, analyze_content_pool
from src.tools.bidding import set_bid_strategy
from src.tools.creative_generator import load_strategy
from src.tools.performance import evaluate_performance
from src.tools.targeting import generate_recommendation, match_user_content

if TYPE_CHECKING:
    from src.memory.user_profile import UserProfileStore


def get_all_tools(user_profile_store: "UserProfileStore | None" = None) -> list:
    """Return all ad tools for the LangGraph executor.

    Args:
        user_profile_store: Optional L2 user-profile FAISS store. When provided,
            a ``find_similar_users`` tool is added so the ReAct agent can query
            the index during recommendation reasoning.
    """
    tools = [
        analyze_audience,
        analyze_content_pool,
        match_user_content,
        generate_recommendation,
        load_strategy,
        set_bid_strategy,
        evaluate_performance,
    ]
    if user_profile_store is not None:
        # Imported here to avoid a circular import: user_retrieval → user_profile
        # is fine, but we keep the top-level `src.tools` import path clean.
        from src.tools.user_retrieval import build_find_similar_users_tool

        tools.append(build_find_similar_users_tool(user_profile_store))
    return tools


__all__ = [
    "analyze_audience",
    "analyze_content_pool",
    "match_user_content",
    "generate_recommendation",
    "load_strategy",
    "set_bid_strategy",
    "evaluate_performance",
    "get_all_tools",
]
