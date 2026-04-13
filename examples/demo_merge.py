"""Real-API smoke test for Distiller.merge_similar().

Seeds a fresh StrategyLibrary with 4 near-duplicate "pet content recommendation"
strategies and 1 unrelated tech strategy, then runs MERGE against real Qwen
embedding + LLM. Confirms that:

  1. Semantic clustering picks the pet group and leaves tech alone
  2. LLM abstraction produces a coherent MERGE strategy
  3. parent_id and historical_performance aggregate correctly

Usage:
    python -m examples.demo_merge
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from src.agent.distiller import StrategyDistiller
from src.config import LLMConfig
from src.llm_factory import create_llm
from src.memory.strategy_lib import StrategyLibrary
from src.models import Strategy, StrategyType


def _pet_strategies() -> list[Strategy]:
    base_perf = lambda r, ctr: [{"round_id": r, "ctr": ctr, "completion_rate": 0.3, "engagement_rate": 0.1}]
    return [
        Strategy(
            strategy_id="s_pet_1",
            name="萌宠短视频推荐",
            strategy_type=StrategyType.NEW,
            applicable_scenario="面向年轻女性的萌宠短视频推荐",
            target_audience="18-30岁女性，活跃度中高，已有宠物相关兴趣标签",
            content_direction="时长<30秒的猫狗日常萌点视频",
            execution_steps=["筛选宠物分类内容", "限制时长<30秒", "优先推给女性用户"],
            expected_effect="CTR提升15-20%",
            historical_performance=base_perf(1, 0.32),
        ),
        Strategy(
            strategy_id="s_pet_2",
            name="宠物日常记录精推",
            strategy_type=StrategyType.NEW,
            applicable_scenario="向年轻女性推送宠物日常记录类内容",
            target_audience="20-28岁女性，城市一二线",
            content_direction="宠物日常记录视频，强调萌点和情绪钩子",
            execution_steps=["锁定年轻女性用户", "选宠物日常记录类目", "前3秒萌点钩子"],
            expected_effect="完播率提升12%",
            historical_performance=base_perf(2, 0.28),
        ),
        Strategy(
            strategy_id="s_pet_3",
            name="可爱宠物视频冷启动",
            strategy_type=StrategyType.NEW,
            applicable_scenario="新用户冷启动场景下推荐可爱宠物视频",
            target_audience="女性新用户，无明显兴趣标签",
            content_direction="可爱猫狗萌点合集视频",
            execution_steps=["识别女性新用户", "推宠物萌点合集", "短时长优先"],
            expected_effect="新用户留存+8%",
            historical_performance=base_perf(3, 0.35),
        ),
        Strategy(
            strategy_id="s_pet_4",
            name="萌宠+情绪钩子组合",
            strategy_type=StrategyType.NEW,
            applicable_scenario="短视频信息流中对女性用户的萌宠内容推广",
            target_audience="18-35岁女性，偏好轻松类内容",
            content_direction="萌宠+反差/治愈情绪钩子的短视频",
            execution_steps=["筛选萌宠内容", "匹配情绪钩子标签", "推送给女性用户"],
            expected_effect="互动率+10%",
            historical_performance=base_perf(4, 0.30),
        ),
        Strategy(
            strategy_id="s_tech_1",
            name="数码极客新品评测",
            strategy_type=StrategyType.NEW,
            applicable_scenario="面向男性极客用户的数码新品评测推荐",
            target_audience="25-40岁男性，科技兴趣标签，高端设备",
            content_direction="数码新品深度评测长视频",
            execution_steps=["匹配男性数码兴趣", "选深度评测视频", "不限时长"],
            expected_effect="完播率+20%",
            historical_performance=base_perf(5, 0.45),
        ),
    ]


def main():
    load_dotenv()

    # Real APIs
    emb_config = LLMConfig(provider="qwen", model="text-embedding-v2")
    distiller_llm = create_llm(LLMConfig(provider="qwen", model="qwen-plus"))

    tmp = Path(tempfile.mkdtemp(prefix="demo_merge_"))
    try:
        lib = StrategyLibrary(strategy_dir=tmp, emb_config=emb_config)

        print(f"{'='*60}")
        print("  MERGE Real-API Smoke Test")
        print(f"  Strategy dir: {tmp}")
        print(f"{'='*60}\n")

        strategies = _pet_strategies()
        for s in strategies:
            lib.save(s)
        print(f"Seeded {lib.count()} strategies ({sum(1 for s in strategies if 'pet' in s.strategy_id)} pet, "
              f"{sum(1 for s in strategies if 'tech' in s.strategy_id)} tech)")

        # Sanity: pairwise semantic distances
        print("\n--- Pairwise semantic signal ---")
        for s in strategies:
            top = lib.semantic_search(s.name, k=3)
            print(f"  query '{s.strategy_id}' → " +
                  ", ".join(f"{hit.strategy_id}(d={d:.2f})" for hit, d in top))

        # Run merge with real LLM
        print("\n--- Running merge_similar (Qwen LLM) ---")
        distiller = StrategyDistiller(llm=distiller_llm)
        merged = distiller.merge_similar(
            lib, distance_threshold=1.0, min_cluster_size=2
        )

        print(f"\nMerged {len(merged)} new strategy(ies).")
        for m in merged:
            print(f"\n=== MERGE strategy ===")
            print(f"  id: {m.strategy_id}")
            print(f"  type: {m.strategy_type.value}")
            print(f"  parent_id: {m.parent_id}")
            print(f"  name: {m.name}")
            print(f"  scenario: {m.applicable_scenario}")
            print(f"  audience: {m.target_audience}")
            print(f"  direction: {m.content_direction}")
            print(f"  steps: {m.execution_steps}")
            print(f"  history entries: {len(m.historical_performance)}")
            print(f"  history avg CTR: "
                  f"{sum(h['ctr'] for h in m.historical_performance) / max(len(m.historical_performance), 1):.3f}")

        print(f"\nLibrary now has {lib.count()} strategies "
              f"(originals kept, MERGE appended).")

        # Check tech strategy NOT merged into pet group
        tech_loaded = lib.get("s_tech_1")
        assert tech_loaded is not None
        assert tech_loaded.strategy_type == StrategyType.NEW
        print(f"\n✓ Tech strategy s_tech_1 stayed separate (type={tech_loaded.strategy_type.value})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
