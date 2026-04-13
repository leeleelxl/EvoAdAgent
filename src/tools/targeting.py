"""Targeting tools — user-content matching and recommendation decisions."""

from __future__ import annotations

import json

from langchain_core.tools import tool


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
        if cat in interests:
            score += 3
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
        确认消息（JSON 格式）
    """
    return json.dumps(
        {"user_id": user_id, "item_id": item_id, "targeting_reason": targeting_reason},
        ensure_ascii=False,
    )
