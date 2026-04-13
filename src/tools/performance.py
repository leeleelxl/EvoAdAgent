"""Performance evaluation tools — CTR/CVR/engagement metrics from campaign results."""

from __future__ import annotations

import json

from langchain_core.tools import tool


@tool
def evaluate_performance(campaign_result_json: str) -> str:
    """评估一次投放的效果指标（CTR、完播率、互动率、综合评分）。

    Args:
        campaign_result_json: 投放结果的 JSON，需包含 total_impressions / clicks /
            completes / likes / shares 字段。

    Returns:
        Markdown 格式的效果报告
    """
    try:
        r = json.loads(campaign_result_json)
    except json.JSONDecodeError:
        return "解析错误：请传入有效的 JSON 格式投放结果"

    impressions = max(int(r.get("total_impressions", 0)), 1)
    clicks = int(r.get("clicks", 0))
    completes = int(r.get("completes", 0))
    likes = int(r.get("likes", 0))
    shares = int(r.get("shares", 0))

    ctr = clicks / impressions
    completion_rate = completes / impressions
    engagement_rate = (likes + shares) / impressions

    # Weighted composite score — completion and engagement valued over raw clicks,
    # because high-CTR-low-engagement campaigns often mask a content quality gap.
    composite = round(0.2 * ctr + 0.4 * completion_rate + 0.4 * engagement_rate, 4)

    verdict = "优秀" if composite >= 0.25 else "良好" if composite >= 0.15 else "待优化"

    return (
        f"## 投放效果评估\n\n"
        f"- 曝光 {impressions} | 点击 {clicks} | 完播 {completes} | 点赞 {likes} | 分享 {shares}\n"
        f"- **CTR:** {ctr:.2%}\n"
        f"- **完播率:** {completion_rate:.2%}\n"
        f"- **互动率:** {engagement_rate:.2%}\n"
        f"- **综合评分:** {composite} ({verdict})\n"
    )
