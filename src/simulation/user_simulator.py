"""LLM-based user behavior simulator, inspired by RecAgent.

Design: the LLM estimates interaction *probabilities* (what it's good at),
and Python does the Bernoulli sampling (what LLMs are bad at — they tend
to over-agree and produce saturated 100% click rates).

The RNG is deterministically seeded per (user_id, item_id) so the same
(user, item) pair always gets the same outcome — reproducible evaluation.
"""

from __future__ import annotations

import hashlib
import json
import random

from langchain_openai import ChatOpenAI

from src.models import ActionType, ContentItem, RecommendAction, UserFeedback, UserProfile

SIMULATOR_PROMPT = """你是一个用户行为模拟器。你不做二元决定，而是**估计该用户与该内容交互的基准概率**。最终的 click/like/share 结果由外部按这些概率采样，你只需给出合理的概率估计。

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

## 基准概率参考（快手平台分布）
| 匹配度 | click_prob | complete_prob(给定点击) | like_prob(给定观看) | share_prob(给定观看) |
|---|---|---|---|---|
| 强匹配（兴趣完全命中） | 0.55-0.75 | 0.35-0.55 | 0.20-0.40 | 0.05-0.15 |
| 中匹配（部分相关） | 0.20-0.40 | 0.15-0.30 | 0.05-0.15 | 0.01-0.05 |
| 弱匹配（擦边） | 0.05-0.15 | 0.05-0.15 | 0.01-0.05 | 0.00-0.02 |
| 不匹配 | 0.01-0.05 | 0.01-0.05 | 0.00-0.01 | 0.00-0.01 |

内容时长影响 complete_prob：<30s→上限，30-120s→中，>120s→下限。

## 输出（严格 JSON）
```json
{{
    "click_prob": 0.0-1.0,
    "complete_prob": 0.0-1.0,
    "like_prob": 0.0-1.0,
    "share_prob": 0.0-1.0,
    "match_level": "strong|medium|weak|mismatch",
    "reason": "简短说明判断依据（为什么是这个匹配度）"
}}
```

不要输出 clicked/liked 等布尔字段，只输出概率。只输出 JSON。"""


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
        return self._parse_and_sample(response.content, user.user_id, content.item_id)

    def simulate_batch(self, actions: list[RecommendAction]) -> list[UserFeedback]:
        """Simulate a batch of user responses."""
        return [self.simulate(action) for action in actions]

    @staticmethod
    def _seed_rng(user_id: str, item_id: str) -> random.Random:
        """Deterministic per-pair RNG — same (user, item) → same outcome.

        Enables reproducible evaluation and fair before/after strategy comparisons.
        """
        key = f"{user_id}|{item_id}".encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
        return random.Random(seed)

    @classmethod
    def _parse_and_sample(cls, text: str, user_id: str, item_id: str) -> UserFeedback:
        """Extract probabilities from the LLM response and Bernoulli-sample an outcome."""
        try:
            text = text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            data = json.loads(text.strip())

            # Clamp to avoid degenerate 0/1 outputs from over-confident LLMs.
            def _clip(p: float) -> float:
                return max(0.0, min(1.0, float(p)))

            click_prob = _clip(data.get("click_prob", 0.1))
            complete_prob = _clip(data.get("complete_prob", 0.2))
            like_prob = _clip(data.get("like_prob", 0.05))
            share_prob = _clip(data.get("share_prob", 0.02))
            reason = data.get("reason", "")
            match_level = data.get("match_level", "")
        except (json.JSONDecodeError, ValueError, KeyError):
            return UserFeedback(
                user_id=user_id,
                item_id=item_id,
                action=ActionType.SKIP,
                watch_ratio=0.0,
                reason="parse error fallback",
            )

        return cls._sample_outcome(
            user_id, item_id, click_prob, complete_prob, like_prob, share_prob,
            reason=f"[{match_level}] {reason}" if match_level else reason,
        )

    @classmethod
    def _sample_outcome(
        cls,
        user_id: str,
        item_id: str,
        click_prob: float,
        complete_prob: float,
        like_prob: float,
        share_prob: float,
        reason: str = "",
    ) -> UserFeedback:
        """Bernoulli-sample a concrete user action from the LLM's probabilities."""
        rng = cls._seed_rng(user_id, item_id)

        clicked = rng.random() < click_prob
        if not clicked:
            return UserFeedback(
                user_id=user_id,
                item_id=item_id,
                action=ActionType.SKIP,
                watch_ratio=0.0,
                reason=reason,
            )

        # Given click, sample watch_ratio: centered near complete_prob with
        # plausible spread. Beta distribution gives a smooth [0,1] distribution.
        alpha = 2 + complete_prob * 6
        beta_param = 2 + (1 - complete_prob) * 6
        watch_ratio = rng.betavariate(alpha, beta_param)

        # Only sample engagement if user watched enough to form an opinion.
        liked = watch_ratio >= 0.3 and rng.random() < like_prob
        shared = watch_ratio >= 0.3 and rng.random() < share_prob

        if shared:
            action = ActionType.SHARE
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
            watch_ratio=round(watch_ratio, 3),
            reason=reason,
        )
