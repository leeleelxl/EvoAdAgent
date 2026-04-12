"""Ad Reflector — Reflexion-based post-campaign analysis.

After each campaign round, the reflector analyzes what worked, what failed,
and why. It produces structured reflections that feed into strategy distillation.
Inspired by Reflexion (NeurIPS 2023) — verbal reinforcement learning.
"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.models import AdReflection, CampaignResult

REFLECTOR_PROMPT = """你是一个广告投放效果分析专家。你需要分析一次推荐投放的结果，找出成功和失败的原因。

## 本轮投放数据
- 轮次: {round_id}
- 总曝光: {impressions}
- 点击数: {clicks} (CTR: {ctr:.2%})
- 完播数: {completes} (完播率: {completion_rate:.2%})
- 点赞数: {likes}
- 分享数: {shares}
- 互动率: {engagement_rate:.2%}
- 使用策略: {strategy_used}

## 详细投放轨迹
{trajectory}

## 历史趋势（最近几轮）
{history}

## 你的任务
分析这次投放的效果，输出结构化的反思报告，格式为 JSON：

```json
{{
    "what_worked": ["有效的做法1", "有效的做法2"],
    "what_failed": ["失败的做法1", "失败的做法2"],
    "root_causes": ["失败的根本原因1", "根本原因2"],
    "improvement_suggestions": ["改进建议1", "改进建议2", "改进建议3"],
    "key_insight": "本轮最关键的一条洞察（一句话）"
}}
```

注意：
- what_worked: 哪些用户-内容匹配是成功的？为什么？
- what_failed: 哪些推荐被跳过了？为什么用户不感兴趣？
- root_causes: 失败的深层原因（人群定向太宽？内容不匹配？时机不对？）
- improvement_suggestions: 下一轮应该怎么改进？要具体可执行
- key_insight: 浓缩成一句话的核心发现

只输出 JSON，不要其他内容。"""


class AdReflector:
    """Analyzes campaign results and produces structured reflections."""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def reflect(
        self,
        result: CampaignResult,
        history: list[dict] | None = None,
    ) -> AdReflection:
        """Reflect on a campaign round's results."""
        trajectory_text = self._format_trajectory(result.trajectory)
        history_text = self._format_history(history or [])

        prompt = REFLECTOR_PROMPT.format(
            round_id=result.round_id,
            impressions=result.total_impressions,
            clicks=result.clicks,
            ctr=result.ctr,
            completes=result.completes,
            completion_rate=result.completion_rate,
            likes=result.likes,
            shares=result.shares,
            engagement_rate=result.engagement_rate,
            strategy_used=result.strategy_used or "无（首次投放）",
            trajectory=trajectory_text,
            history=history_text,
        )

        messages = [
            SystemMessage(content="你是一个广告投放效果分析专家。"),
            HumanMessage(content=prompt),
        ]
        response = self.llm.invoke(messages)
        return self._parse_response(response.content, result)

    @staticmethod
    def _format_trajectory(trajectory: list[dict]) -> str:
        lines = []
        for t in trajectory[:20]:  # Limit to avoid context overflow
            action_emoji = {"click": "✅", "skip": "❌", "like": "👍", "share": "🔄",
                          "comment": "💬", "complete_play": "🎬"}.get(t.get("feedback_action", ""), "❓")
            lines.append(
                f"  {action_emoji} 用户{t['user_id']} → 内容{t['item_id']}: "
                f"{t['feedback_action']}(观看{t['watch_ratio']:.0%}) "
                f"| 推荐理由: {t['targeting_reason'][:50]} "
                f"| 用户反馈: {t.get('feedback_reason', '')[:50]}"
            )
        if len(trajectory) > 20:
            lines.append(f"  ... 还有 {len(trajectory) - 20} 条记录")
        return "\n".join(lines)

    @staticmethod
    def _format_history(history: list[dict]) -> str:
        if not history:
            return "暂无历史数据（首次投放）"
        lines = []
        for h in history[-5:]:
            lines.append(
                f"  Round {h['round_id']}: CTR={h.get('ctr', 0):.2%}, "
                f"完播率={h.get('completion_rate', 0):.2%}, "
                f"互动率={h.get('engagement_rate', 0):.2%}, "
                f"策略={h.get('strategy_used', '无')}"
            )
        return "\n".join(lines)

    @staticmethod
    def _parse_response(text: str, result: CampaignResult) -> AdReflection:
        try:
            text = text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            data = json.loads(text.strip())
            return AdReflection(
                round_id=result.round_id,
                campaign_result=result,
                what_worked=data.get("what_worked", []),
                what_failed=data.get("what_failed", []),
                root_causes=data.get("root_causes", []),
                improvement_suggestions=data.get("improvement_suggestions", []),
                key_insight=data.get("key_insight", ""),
            )
        except (json.JSONDecodeError, KeyError):
            return AdReflection(
                round_id=result.round_id,
                campaign_result=result,
                what_worked=[],
                what_failed=["反思报告解析失败"],
                root_causes=["LLM 输出格式异常"],
                improvement_suggestions=["重试反思"],
                key_insight="反思解析失败，需要重试",
            )
