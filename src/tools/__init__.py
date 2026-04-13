"""Ad tools — five specialized tool modules for the LangGraph executor.

Each tool is a LangChain ``@tool``-decorated function. ``get_all_tools()``
aggregates them for the executor's ToolNode.
"""

from src.tools.audience_analyzer import analyze_audience, analyze_content_pool
from src.tools.bidding import set_bid_strategy
from src.tools.creative_generator import load_strategy
from src.tools.performance import evaluate_performance
from src.tools.targeting import generate_recommendation, match_user_content


def get_all_tools() -> list:
    """Return all ad tools for the LangGraph executor."""
    return [
        analyze_audience,
        analyze_content_pool,
        match_user_content,
        generate_recommendation,
        load_strategy,
        set_bid_strategy,
        evaluate_performance,
    ]


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
