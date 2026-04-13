"""Audience analysis tools — user demographic and content pool profiling."""

from __future__ import annotations

import json

from langchain_core.tools import tool


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
    genders: dict[str, int] = {}
    ages: dict[str, int] = {}
    city_levels: dict[str, int] = {}
    all_interests: dict[str, int] = {}
    for u in users:
        g = u.get("gender", "unknown")
        genders[g] = genders.get(g, 0) + 1
        a = u.get("age_range", "unknown")
        ages[a] = ages.get(a, 0) + 1
        cl = u.get("city_level", "unknown")
        city_levels[cl] = city_levels.get(cl, 0) + 1
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
    categories: dict[str, int] = {}
    all_tags: dict[str, int] = {}
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
