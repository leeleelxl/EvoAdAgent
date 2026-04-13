"""Tests for StrategyDistiller — focus on MERGE logic (NEW/REFINE path
requires live LLM and is covered by examples/demo_evolution.py)."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from src.agent.distiller import StrategyDistiller
from src.config import LLMConfig
from src.memory.strategy_lib import StrategyLibrary
from src.models import Strategy, StrategyType


class _FakeEmbedder:
    """Deterministic embedder. Strategies sharing keywords get close vectors."""

    def __init__(self, dim: int = 32):
        self.dim = dim

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t).tolist() for t in texts]

    def _vec(self, text: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self.dim).astype("float32")
        v /= np.linalg.norm(v) + 1e-9
        return v


class _FakeLLM:
    """Scripted LLM — returns a pre-built response for each .invoke() call."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[list] = []

    def invoke(self, messages):
        self.calls.append(messages)

        class _Resp:
            def __init__(self, content):
                self.content = content

        return _Resp(self._responses.pop(0))


def _merge_response(name: str = "抽象元策略", action: str = "merge") -> str:
    return json.dumps(
        {
            "action": action,
            "strategy": {
                "name": name,
                "applicable_scenario": "跨场景的通用推荐",
                "target_audience": "广泛人群",
                "content_direction": "通用内容",
                "execution_steps": ["通用步骤1", "通用步骤2"],
                "expected_effect": "CTR 和互动率均衡提升",
            },
        },
        ensure_ascii=False,
    )


def _make_strategy(
    sid: str,
    name: str,
    scenario: str = "通用",
    audience: str = "通用",
    direction: str = "通用",
) -> Strategy:
    return Strategy(
        strategy_id=sid,
        name=name,
        strategy_type=StrategyType.NEW,
        applicable_scenario=scenario,
        target_audience=audience,
        content_direction=direction,
        execution_steps=["step1"],
        expected_effect="+5%",
        historical_performance=[{"round_id": 1, "ctr": 0.1}],
    )


@pytest.fixture()
def lib_with_fake_embedder(tmp_path):
    lib = StrategyLibrary(
        strategy_dir=tmp_path / "strategies",
        emb_config=LLMConfig(provider="openai", model="fake"),
    )
    lib._embedder = _FakeEmbedder(dim=32)
    return lib


class TestMergeNoIndex:
    def test_merge_without_vector_index_returns_empty(self, tmp_path):
        lib = StrategyLibrary(strategy_dir=tmp_path / "strategies")  # no emb_config
        distiller = StrategyDistiller(llm=_FakeLLM([]))

        assert distiller.merge_similar(lib) == []


class TestMergeClustering:
    def test_no_similar_strategies_no_merge(self, lib_with_fake_embedder):
        lib = lib_with_fake_embedder
        lib.save(_make_strategy("s1", "策略A", scenario="宠物"))
        lib.save(_make_strategy("s2", "策略B", scenario="金融"))
        lib.save(_make_strategy("s3", "策略C", scenario="科技"))

        distiller = StrategyDistiller(llm=_FakeLLM([]))
        merged = distiller.merge_similar(
            lib, distance_threshold=0.001, min_cluster_size=2
        )
        assert merged == []

    def test_identical_signatures_trigger_merge(self, lib_with_fake_embedder):
        lib = lib_with_fake_embedder
        # Two strategies with identical signatures → L2 distance ≈ 0
        lib.save(_make_strategy("s1", "策略A", "同场景", "同人群", "同方向"))
        lib.save(_make_strategy("s2", "策略A", "同场景", "同人群", "同方向"))
        lib.save(_make_strategy("s3", "无关策略", "完全不同", "完全不同", "完全不同"))

        distiller = StrategyDistiller(llm=_FakeLLM([_merge_response()]))
        merged = distiller.merge_similar(
            lib, distance_threshold=0.01, min_cluster_size=2
        )

        assert len(merged) == 1
        m = merged[0]
        assert m.strategy_type == StrategyType.MERGE
        assert m.name == "抽象元策略"
        assert m.parent_id in ("s1", "s2")

    def test_merged_strategy_saved_to_library(self, lib_with_fake_embedder):
        lib = lib_with_fake_embedder
        lib.save(_make_strategy("s1", "策略A", "同", "同", "同"))
        lib.save(_make_strategy("s2", "策略A", "同", "同", "同"))

        distiller = StrategyDistiller(llm=_FakeLLM([_merge_response()]))
        before = lib.count()
        distiller.merge_similar(lib, distance_threshold=0.01, min_cluster_size=2)

        assert lib.count() == before + 1  # original kept + merged added

    def test_merged_strategy_aggregates_historical_performance(
        self, lib_with_fake_embedder
    ):
        lib = lib_with_fake_embedder
        s1 = _make_strategy("s1", "策略A", "同", "同", "同")
        s1.historical_performance = [{"round_id": 1, "ctr": 0.1}]
        s2 = _make_strategy("s2", "策略A", "同", "同", "同")
        s2.historical_performance = [{"round_id": 2, "ctr": 0.2}]
        lib.save(s1)
        lib.save(s2)

        distiller = StrategyDistiller(llm=_FakeLLM([_merge_response()]))
        merged = distiller.merge_similar(
            lib, distance_threshold=0.01, min_cluster_size=2
        )
        assert len(merged[0].historical_performance) == 2

    def test_llm_skip_action_produces_no_merge(self, lib_with_fake_embedder):
        lib = lib_with_fake_embedder
        lib.save(_make_strategy("s1", "策略A", "同", "同", "同"))
        lib.save(_make_strategy("s2", "策略A", "同", "同", "同"))

        distiller = StrategyDistiller(
            llm=_FakeLLM([_merge_response(action="skip")])
        )
        merged = distiller.merge_similar(
            lib, distance_threshold=0.01, min_cluster_size=2
        )
        assert merged == []

    def test_each_strategy_in_at_most_one_cluster(self, lib_with_fake_embedder):
        lib = lib_with_fake_embedder
        # Three strategies with identical signature → all one cluster
        for i in range(3):
            lib.save(_make_strategy(f"s{i}", "同策略", "同", "同", "同"))

        distiller = StrategyDistiller(llm=_FakeLLM([_merge_response()]))
        merged = distiller.merge_similar(
            lib, distance_threshold=0.01, min_cluster_size=2
        )
        # Should produce exactly one merged strategy, not 3 overlapping ones
        assert len(merged) == 1

    def test_min_cluster_size_enforced(self, lib_with_fake_embedder):
        lib = lib_with_fake_embedder
        lib.save(_make_strategy("s1", "策略A", "同", "同", "同"))
        lib.save(_make_strategy("s2", "策略A", "同", "同", "同"))

        distiller = StrategyDistiller(llm=_FakeLLM([]))
        # min_cluster_size=3 → no merge (only 2 identical)
        merged = distiller.merge_similar(
            lib, distance_threshold=0.01, min_cluster_size=3
        )
        assert merged == []


class TestRefineValidation:
    def test_refine_with_valid_parent_bumps_version(self):
        strat = _make_strategy("child", "child_name")
        strat.strategy_type = StrategyType.REFINE
        strat.parent_id = "parent"
        existing = [{"strategy_id": "parent", "name": "p", "version": 2}]
        out = StrategyDistiller._validate_refine_link(strat, existing)
        assert out.strategy_type == StrategyType.REFINE
        assert out.parent_id == "parent"
        assert out.version == 3

    def test_refine_with_missing_parent_demotes_to_new(self):
        strat = _make_strategy("orphan", "orphan_name")
        strat.strategy_type = StrategyType.REFINE
        strat.parent_id = "does_not_exist"
        existing = [{"strategy_id": "parent", "name": "p", "version": 1}]
        out = StrategyDistiller._validate_refine_link(strat, existing)
        assert out.strategy_type == StrategyType.NEW
        assert out.parent_id is None
        assert out.version == 1

    def test_refine_with_null_parent_demotes_to_new(self):
        strat = _make_strategy("orphan", "orphan_name")
        strat.strategy_type = StrategyType.REFINE
        strat.parent_id = None
        out = StrategyDistiller._validate_refine_link(strat, [])
        assert out.strategy_type == StrategyType.NEW

    def test_refine_with_empty_existing_demotes_to_new(self):
        strat = _make_strategy("orphan", "orphan_name")
        strat.strategy_type = StrategyType.REFINE
        strat.parent_id = "anything"
        out = StrategyDistiller._validate_refine_link(strat, None)
        assert out.strategy_type == StrategyType.NEW

    def test_new_strategy_unchanged(self):
        strat = _make_strategy("s1", "n")
        strat.strategy_type = StrategyType.NEW
        out = StrategyDistiller._validate_refine_link(strat, None)
        assert out.strategy_type == StrategyType.NEW
        assert out is strat


class TestMergeParsing:
    def test_parse_json_in_code_fence(self):
        cluster = [_make_strategy("s1", "A"), _make_strategy("s2", "B")]
        response = f"```json\n{_merge_response()}\n```"
        parsed = StrategyDistiller._parse_merge_response(response, cluster)
        assert parsed is not None
        assert parsed.strategy_type == StrategyType.MERGE

    def test_parse_invalid_json_returns_none(self):
        cluster = [_make_strategy("s1", "A")]
        assert StrategyDistiller._parse_merge_response("not json", cluster) is None

    def test_parse_merged_has_lineage_to_first_cluster_member(self):
        cluster = [_make_strategy("s1", "A"), _make_strategy("s2", "B")]
        parsed = StrategyDistiller._parse_merge_response(_merge_response(), cluster)
        assert parsed.parent_id == "s1"
