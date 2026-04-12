"""Strategy Distiller — extracts reusable strategies from reflections.

Three distillation modes:
- NEW: Extract a brand-new strategy from a novel insight
- REFINE: Improve an existing strategy based on failure analysis
- MERGE: Combine similar strategies into a more general one
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.models import AdReflection, CampaignResult, Strategy, StrategyType

DISTILL_PROMPT = """你是一个推荐策略提炼专家。基于投放反思，你需要提炼出可复用的推荐策略。

## 投放反思报告
- 轮次: {round_id}
- CTR: {ctr:.2%} | 完播率: {completion_rate:.2%} | 互动率: {engagement_rate:.2%}
- 成功经验: {what_worked}
- 失败经验: {what_failed}
- 根本原因: {root_causes}
- 改进建议: {improvement_suggestions}
- 核心洞察: {key_insight}

## 现有策略库
{existing_strategies}

## 你的任务
判断是否应该提炼新策略，以及用什么方式：

1. **NEW** — 如果发现了全新的有效模式，创建新策略
2. **REFINE** — 如果现有策略可以改进，指出要改进哪个策略
3. **SKIP** — 如果没有足够信息提炼策略，跳过

输出 JSON：
```json
{{
    "action": "new|refine|skip",
    "refine_target_id": "策略ID（仅 refine 时填写）",
    "strategy": {{
        "name": "策略名称（简洁有力）",
        "applicable_scenario": "适用场景描述",
        "target_audience": "目标人群特征",
        "content_direction": "推荐什么类型的内容",
        "execution_steps": ["步骤1", "步骤2", "步骤3"],
        "expected_effect": "预期效果"
    }}
}}
```

只在有明确可复用的模式时才创建策略。不要为了创建而创建。
只输出 JSON，不要其他内容。"""


class StrategyDistiller:
    """Distills reusable strategies from campaign reflections."""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def distill(
        self,
        reflection: AdReflection,
        existing_strategies: list[dict] | None = None,
    ) -> Strategy | None:
        """Try to distill a strategy from a reflection. Returns None if SKIP."""
        existing_text = "暂无策略" if not existing_strategies else "\n".join(
            f"- [{s['strategy_id']}] {s['name']}: {s['applicable_scenario']}"
            for s in (existing_strategies or [])
        )

        result = reflection.campaign_result
        prompt = DISTILL_PROMPT.format(
            round_id=reflection.round_id,
            ctr=result.ctr,
            completion_rate=result.completion_rate,
            engagement_rate=result.engagement_rate,
            what_worked="; ".join(reflection.what_worked),
            what_failed="; ".join(reflection.what_failed),
            root_causes="; ".join(reflection.root_causes),
            improvement_suggestions="; ".join(reflection.improvement_suggestions),
            key_insight=reflection.key_insight,
            existing_strategies=existing_text,
        )

        messages = [
            SystemMessage(content="你是推荐策略提炼专家。"),
            HumanMessage(content=prompt),
        ]
        response = self.llm.invoke(messages)
        return self._parse_response(response.content, reflection)

    @staticmethod
    def _parse_response(text: str, reflection: AdReflection) -> Strategy | None:
        try:
            text = text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            data = json.loads(text.strip())

            action = data.get("action", "skip")
            if action == "skip":
                return None

            s = data.get("strategy", {})
            strategy_type = StrategyType.NEW if action == "new" else StrategyType.REFINE
            strategy_id = f"strat_{uuid.uuid4().hex[:8]}"

            result = reflection.campaign_result
            return Strategy(
                strategy_id=strategy_id,
                name=s.get("name", "unnamed"),
                strategy_type=strategy_type,
                applicable_scenario=s.get("applicable_scenario", ""),
                target_audience=s.get("target_audience", ""),
                content_direction=s.get("content_direction", ""),
                execution_steps=s.get("execution_steps", []),
                expected_effect=s.get("expected_effect", ""),
                historical_performance=[{
                    "round_id": result.round_id,
                    "ctr": result.ctr,
                    "completion_rate": result.completion_rate,
                    "engagement_rate": result.engagement_rate,
                }],
                parent_id=data.get("refine_target_id"),
            )
        except (json.JSONDecodeError, KeyError):
            return None
