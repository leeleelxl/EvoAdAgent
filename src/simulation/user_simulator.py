"""LLM-based user behavior simulator, inspired by RecAgent.

Uses LLM to simulate how different user personas would react to content
recommendations. The simulator produces realistic click/skip/like/share
decisions based on user profiles and content features.
"""

from __future__ import annotations

import json
import random

from langchain_openai import ChatOpenAI

from src.models import ActionType, ContentItem, RecommendAction, UserFeedback, UserProfile

SIMULATOR_PROMPT = """你是一个用户行为模拟器。你需要模拟一个真实用户在短视频/信息流平台上看到推荐内容后的**完整行为链**。

## 用户画像
- 性别: {gender}
- 年龄段: {age_range}
- 地区: {province} {city} ({city_level})
- 兴趣偏好: {interests}
- 活跃度: {active_degree}
- 设备档次: {device_price}

## 推荐内容
- 标题: {caption}
- 话题标签: {topic_tags}
- 分类: {category_l1} > {category_l2} > {category_l3}
- 时长: {duration}秒

## 推荐理由
{targeting_reason}

## 你的任务
模拟这个用户看到该推荐后的**三阶段行为链**：

### 阶段1：是否点击？
用户是否会被标题/封面吸引而点击？考虑兴趣匹配度。

### 阶段2：观看多少？
如果点击了，用户会看多少比例？考虑内容质量和时长。
- 短视频(<30秒)：匹配的用户完播率约 40-60%
- 中等视频(30-120秒)：完播率约 20-40%
- 长视频(>120秒)：完播率约 10-25%

### 阶段3：是否互动？
如果看了超过 50%，用户是否会点赞/评论/分享？
- 兴趣强匹配 + 内容实用 → 点赞概率 30-50%
- 内容有社交分享价值 → 分享概率 5-15%
- 内容引发讨论 → 评论概率 5-10%

请输出 JSON：
```json
{{
    "clicked": true/false,
    "watch_ratio": 0.0-1.0,
    "liked": true/false,
    "shared": true/false,
    "commented": true/false,
    "reason": "简短解释用户的整体行为逻辑"
}}
```

参考基准（基于快手平台数据分布）：
- 兴趣强匹配：点击率 ~70%，完播率 ~45%，点赞率 ~25%
- 兴趣弱匹配：点击率 ~30%，完播率 ~15%，点赞率 ~5%
- 兴趣不匹配：点击率 ~10%，完播率 ~3%，点赞率 ~1%

只输出 JSON，不要其他内容。"""


class UserSimulator:
    """Simulates user responses to content recommendations using LLM."""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def simulate(self, action: RecommendAction) -> UserFeedback:
        """Simulate a single user's response to a recommendation."""
        user = action.user
        content = action.content

        prompt = SIMULATOR_PROMPT.format(
            gender=user.gender.value,
            age_range=user.age_range,
            province=user.province,
            city=user.city,
            city_level=user.city_level,
            interests=", ".join(user.interests),
            active_degree=user.active_degree,
            device_price=user.device_price,
            caption=content.caption,
            topic_tags=", ".join(content.topic_tags),
            category_l1=content.category_l1,
            category_l2=content.category_l2,
            category_l3=content.category_l3,
            duration=content.duration_seconds,
            targeting_reason=action.targeting_reason,
        )

        response = self.llm.invoke(prompt)
        return self._parse_response(response.content, user.user_id, content.item_id)

    def simulate_batch(self, actions: list[RecommendAction]) -> list[UserFeedback]:
        """Simulate a batch of user responses."""
        return [self.simulate(action) for action in actions]

    @staticmethod
    def _parse_response(text: str, user_id: str, item_id: str) -> UserFeedback:
        """Parse multi-stage behavior chain into a single primary action."""
        try:
            text = text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            data = json.loads(text.strip())

            clicked = data.get("clicked", False)
            watch_ratio = float(data.get("watch_ratio", 0.0))
            liked = data.get("liked", False)
            shared = data.get("shared", False)
            commented = data.get("commented", False)
            reason = data.get("reason", "")

            # Determine primary action from the behavior chain
            if not clicked:
                action = ActionType.SKIP
                watch_ratio = min(watch_ratio, 0.05)
            elif shared:
                action = ActionType.SHARE
            elif commented:
                action = ActionType.COMMENT
            elif liked:
                action = ActionType.LIKE
            elif watch_ratio >= 0.9:
                action = ActionType.COMPLETE_PLAY
            else:
                action = ActionType.CLICK

            return UserFeedback(
                user_id=user_id,
                item_id=item_id,
                action=action,
                watch_ratio=watch_ratio,
                reason=reason,
            )
        except (json.JSONDecodeError, ValueError, KeyError):
            return UserFeedback(
                user_id=user_id,
                item_id=item_id,
                action=ActionType.SKIP,
                watch_ratio=random.uniform(0.0, 0.1),
                reason="parse error fallback",
            )
