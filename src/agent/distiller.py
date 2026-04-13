"""Strategy Distiller — extracts reusable strategies from reflections.

Three distillation modes:
- NEW: Extract a brand-new strategy from a novel insight
- REFINE: Improve an existing strategy based on failure analysis
- MERGE: Combine semantically similar strategies into a more general one
        (requires StrategyLibrary with FAISS vector index)
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.models import AdReflection, CampaignResult, Strategy, StrategyType

if TYPE_CHECKING:
    from src.memory.strategy_lib import StrategyLibrary

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
2. **REFINE** — 如果现有策略可以改进，**必须从"现有策略库"列表中选一个真实的 strategy_id 填入 refine_target_id**，不可留空、不可编造
3. **SKIP** — 如果没有足够信息提炼策略，跳过

**重要：** 如果策略库为空，只能选 NEW 或 SKIP，不能选 REFINE。
**重要：** REFINE 时 `refine_target_id` 必须是现有策略库中真实存在的 strategy_id。

输出 JSON：
```json
{{
    "action": "new|refine|skip",
    "refine_target_id": "必须是现有策略库中的 strategy_id（仅 refine 时填写）",
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


MERGE_PROMPT = """你是推荐策略抽象专家。以下是一组语义相似的推荐策略，它们可能在描述同一类投放模式的不同变体。

## 相似策略组
{strategies_block}

## 你的任务
抽象出一条更通用的"元策略"，概括这组策略共同的核心规律：
- 名称应比单个策略更抽象、更有泛化性
- 适用场景要覆盖所有原策略的场景
- 执行步骤要提炼共通步骤，去掉细节分支
- 若这些策略没有共同抽象价值（主题差异过大），返回 action=skip

输出 JSON：
```json
{{
    "action": "merge|skip",
    "strategy": {{
        "name": "抽象后的元策略名称",
        "applicable_scenario": "更通用的适用场景",
        "target_audience": "抽象后的人群描述",
        "content_direction": "抽象后的内容方向",
        "execution_steps": ["抽象后的通用步骤1", "步骤2", "步骤3"],
        "expected_effect": "预期泛化效果"
    }}
}}
```

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
        strategy = self._parse_response(response.content, reflection)
        if strategy is None:
            return None
        # Validate REFINE lineage: if LLM said REFINE but didn't point to a
        # real existing strategy, demote to NEW rather than orphan the link.
        return self._validate_refine_link(strategy, existing_strategies)

    @staticmethod
    def _validate_refine_link(
        strategy: Strategy, existing_strategies: list[dict] | None
    ) -> Strategy:
        """Ensure a REFINE strategy actually links to a valid parent. Demote to NEW otherwise."""
        if strategy.strategy_type != StrategyType.REFINE:
            return strategy

        valid_ids = {s["strategy_id"] for s in (existing_strategies or [])}
        if strategy.parent_id and strategy.parent_id in valid_ids:
            # Find the parent's version to bump from.
            parent_version = next(
                (s.get("version", 1) for s in existing_strategies
                 if s["strategy_id"] == strategy.parent_id),
                1,
            )
            strategy.version = parent_version + 1
            return strategy

        # REFINE without a valid parent is meaningless — demote to NEW.
        strategy.strategy_type = StrategyType.NEW
        strategy.parent_id = None
        strategy.version = 1
        return strategy

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

    # --- MERGE ---

    def merge_similar(
        self,
        strategy_lib: "StrategyLibrary",
        distance_threshold: float = 0.5,
        min_cluster_size: int = 2,
    ) -> list[Strategy]:
        """Scan the library for semantically similar strategy clusters and
        abstract each into a more general MERGE strategy.

        Args:
            strategy_lib: A StrategyLibrary with FAISS vector index enabled.
            distance_threshold: Max L2 distance for two strategies to be
                considered "similar enough" to merge.
            min_cluster_size: Minimum strategies per cluster to trigger a merge.

        Returns:
            List of newly created MERGE strategies (also saved to the library).
        """
        if not strategy_lib.has_vector_index:
            return []

        clusters = self._find_similar_clusters(
            strategy_lib, distance_threshold, min_cluster_size
        )
        if not clusters:
            return []

        merged: list[Strategy] = []
        for cluster_ids in clusters:
            cluster = [strategy_lib.get(sid) for sid in cluster_ids]
            cluster = [s for s in cluster if s is not None]
            if len(cluster) < min_cluster_size:
                continue
            merged_strategy = self._merge_cluster(cluster)
            if merged_strategy is not None:
                strategy_lib.save(merged_strategy)
                merged.append(merged_strategy)
        return merged

    def _find_similar_clusters(
        self,
        strategy_lib: "StrategyLibrary",
        distance_threshold: float,
        min_cluster_size: int,
    ) -> list[list[str]]:
        """Greedy clustering: for each unseen strategy, pull its nearest
        neighbors within threshold and form a cluster. Each strategy ends up
        in at most one cluster."""
        from src.memory.strategy_lib import strategy_to_signature

        all_ids = list(strategy_lib._faiss_ids)
        visited: set[str] = set()
        clusters: list[list[str]] = []

        for sid in all_ids:
            if sid in visited:
                continue
            seed = strategy_lib.get(sid)
            if seed is None:
                continue

            neighbors = strategy_lib.semantic_search(
                strategy_to_signature(seed), k=min(len(all_ids), 10)
            )
            cluster = [
                s.strategy_id
                for s, dist in neighbors
                if dist <= distance_threshold and s.strategy_id not in visited
            ]
            if len(cluster) >= min_cluster_size:
                clusters.append(cluster)
                visited.update(cluster)
            else:
                visited.add(sid)
        return clusters

    def _merge_cluster(self, cluster: list[Strategy]) -> Strategy | None:
        """Ask LLM to abstract a cluster into one general strategy."""
        strategies_block = "\n\n".join(
            f"### 策略 {i+1} [{s.strategy_id}] {s.name}\n"
            f"- 场景: {s.applicable_scenario}\n"
            f"- 人群: {s.target_audience}\n"
            f"- 方向: {s.content_direction}\n"
            f"- 步骤: {'; '.join(s.execution_steps)}"
            for i, s in enumerate(cluster)
        )
        prompt = MERGE_PROMPT.format(strategies_block=strategies_block)

        messages = [
            SystemMessage(content="你是推荐策略抽象专家。"),
            HumanMessage(content=prompt),
        ]
        response = self.llm.invoke(messages)
        return self._parse_merge_response(response.content, cluster)

    @staticmethod
    def _parse_merge_response(text: str, cluster: list[Strategy]) -> Strategy | None:
        try:
            text = text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            data = json.loads(text.strip())

            if data.get("action") != "merge":
                return None

            s = data.get("strategy", {})
            strategy_id = f"merged_{uuid.uuid4().hex[:8]}"

            # Aggregate historical performance from all source strategies.
            history: list[dict] = []
            for src in cluster:
                history.extend(src.historical_performance or [])

            return Strategy(
                strategy_id=strategy_id,
                name=s.get("name", "unnamed_merged"),
                strategy_type=StrategyType.MERGE,
                applicable_scenario=s.get("applicable_scenario", ""),
                target_audience=s.get("target_audience", ""),
                content_direction=s.get("content_direction", ""),
                execution_steps=s.get("execution_steps", []),
                expected_effect=s.get("expected_effect", ""),
                historical_performance=history,
                parent_id=cluster[0].strategy_id,  # lineage anchor
            )
        except (json.JSONDecodeError, KeyError):
            return None
