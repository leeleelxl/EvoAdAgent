"""Ad-specific tools for the LangGraph executor.

Each tool is a LangChain Tool that the agent can call during its ReAct loop.
These tools provide the agent with capabilities to analyze audiences,
generate creatives, select targeting, and evaluate performance.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool

from src.models import ContentItem, UserProfile


# --- Tool implementations ---
# These are module-level functions decorated with @tool.
# The executor binds them into a LangGraph ToolNode.


@tool
def analyze_audience(user_profiles_json: str) -> str:
    """分析用户群体画像，提取人群分布特征（性别、年龄、地域、兴趣偏好）。

    Args:
        user_profiles_json: JSON 格式的用户画像列表

    Returns:
        人群分布分析报告
    """
    try:
        users = json.loads(user_profiles_json)
    except json.JSONDecodeError:
        return "解析错误：请传入有效的 JSON 格式用户画像列表"

    if not users:
        return "用户列表为空"

    total = len(users)
    # Gender distribution
    genders = {}
    for u in users:
        g = u.get("gender", "unknown")
        genders[g] = genders.get(g, 0) + 1

    # Age distribution
    ages = {}
    for u in users:
        a = u.get("age_range", "unknown")
        ages[a] = ages.get(a, 0) + 1

    # City level distribution
    city_levels = {}
    for u in users:
        cl = u.get("city_level", "unknown")
        city_levels[cl] = city_levels.get(cl, 0) + 1

    # Interest aggregation
    all_interests = {}
    for u in users:
        for interest in u.get("interests", []):
            all_interests[interest] = all_interests.get(interest, 0) + 1

    top_interests = sorted(all_interests.items(), key=lambda x: -x[1])[:10]

    report = f"## 用户群体分析（共 {total} 人）\n\n"
    report += f"**性别分布:** {json.dumps(genders, ensure_ascii=False)}\n"
    report += f"**年龄分布:** {json.dumps(ages, ensure_ascii=False)}\n"
    report += f"**城市等级:** {json.dumps(city_levels, ensure_ascii=False)}\n"
    report += f"**热门兴趣 Top10:** {json.dumps(dict(top_interests), ensure_ascii=False)}\n"
    return report


@tool
def analyze_content_pool(contents_json: str) -> str:
    """分析可推荐的内容池，提取内容分类分布和特征。

    Args:
        contents_json: JSON 格式的内容列表

    Returns:
        内容池分析报告
    """
    try:
        contents = json.loads(contents_json)
    except json.JSONDecodeError:
        return "解析错误：请传入有效的 JSON 格式内容列表"

    total = len(contents)
    categories = {}
    all_tags = {}
    for c in contents:
        cat = c.get("category_l1", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
        for tag in c.get("topic_tags", []):
            all_tags[tag] = all_tags.get(tag, 0) + 1

    top_tags = sorted(all_tags.items(), key=lambda x: -x[1])[:10]

    report = f"## 内容池分析（共 {total} 条）\n\n"
    report += f"**分类分布:** {json.dumps(categories, ensure_ascii=False)}\n"
    report += f"**热门标签:** {json.dumps(dict(top_tags), ensure_ascii=False)}\n"
    return report


@tool
def match_user_content(user_json: str, contents_json: str) -> str:
    """为单个用户匹配最合适的内容。基于用户兴趣和内容分类/标签做匹配。

    Args:
        user_json: 单个用户的 JSON 画像
        contents_json: 候选内容列表 JSON

    Returns:
        匹配结果和推荐理由
    """
    try:
        user = json.loads(user_json)
        contents = json.loads(contents_json)
    except json.JSONDecodeError:
        return "解析错误"

    interests = set(user.get("interests", []))
    scored = []
    for c in contents:
        score = 0
        cat = c.get("category_l1", "")
        tags = set(c.get("topic_tags", []))
        # Interest-category match
        if cat in interests:
            score += 3
        # Interest-tag overlap
        overlap = interests & tags
        score += len(overlap) * 2
        scored.append((c, score, overlap))

    scored.sort(key=lambda x: -x[1])
    top3 = scored[:3]

    result = f"## 用户 {user.get('user_id', '?')} 的推荐匹配\n\n"
    for c, score, overlap in top3:
        result += (
            f"- **{c['item_id']}** (匹配分={score}): \"{c.get('caption', '')[:50]}\"\n"
            f"  分类={c.get('category_l1', '')} | 匹配兴趣={list(overlap) if overlap else '无直接匹配'}\n"
        )
    return result


@tool
def load_strategy(strategy_info: str) -> str:
    """加载并应用一条推荐策略。

    Args:
        strategy_info: 策略名称或描述

    Returns:
        策略的详细执行步骤
    """
    # This is a placeholder — the actual strategy loading is done in the executor
    # by injecting strategy content into the system prompt.
    return f"策略 '{strategy_info}' 已加载。请按照策略中的执行步骤进行推荐。"


@tool
def generate_recommendation(
    user_id: str,
    item_id: str,
    targeting_reason: str,
) -> str:
    """生成一条最终的推荐决策。

    Args:
        user_id: 目标用户 ID
        item_id: 推荐内容 ID
        targeting_reason: 推荐理由

    Returns:
        确认消息
    """
    return json.dumps(
        {"user_id": user_id, "item_id": item_id, "targeting_reason": targeting_reason},
        ensure_ascii=False,
    )


def get_all_tools() -> list:
    """Return all ad tools for the LangGraph executor."""
    return [
        analyze_audience,
        analyze_content_pool,
        match_user_content,
        load_strategy,
        generate_recommendation,
    ]
