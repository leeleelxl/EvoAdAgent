"""Tests for memory layer: CampaignLog and StrategyLibrary."""

import pytest

from src.memory.campaign_log import CampaignLog
from src.memory.strategy_lib import StrategyLibrary
from src.models import (
    AdReflection,
    CampaignResult,
    Strategy,
    StrategyType,
)


# ============================================================
# CampaignLog Tests
# ============================================================


class TestCampaignLog:
    @pytest.fixture()
    def log(self, tmp_path):
        """Create a CampaignLog backed by a temporary SQLite DB."""
        return CampaignLog(db_path=tmp_path / "test.db")

    def _make_result(self, round_id, impressions=100, clicks=20, completes=10, likes=5, shares=3):
        return CampaignResult(
            round_id=round_id,
            total_impressions=impressions,
            clicks=clicks,
            completes=completes,
            likes=likes,
            shares=shares,
            strategy_used=f"strategy_{round_id}",
            trajectory=[{"step": 1, "action": "test"}],
        )

    def _make_reflection(self, round_id, result):
        return AdReflection(
            round_id=round_id,
            campaign_result=result,
            what_worked=["兴趣匹配准确"],
            what_failed=["低活跃用户未覆盖"],
            root_causes=["用户画像不完整"],
            improvement_suggestions=["引入更多用户特征"],
            key_insight="精准匹配是关键",
        )

    # -- save_campaign & get_recent_campaigns --

    def test_save_and_retrieve_campaign(self, log):
        result = self._make_result(1)
        log.save_campaign(result)

        campaigns = log.get_recent_campaigns(n=10)
        assert len(campaigns) == 1
        assert campaigns[0]["round_id"] == 1
        assert campaigns[0]["total_impressions"] == 100
        assert campaigns[0]["clicks"] == 20
        assert campaigns[0]["strategy_used"] == "strategy_1"

    def test_save_multiple_campaigns(self, log):
        for i in range(1, 6):
            log.save_campaign(self._make_result(i))

        campaigns = log.get_recent_campaigns(n=10)
        assert len(campaigns) == 5

    def test_get_recent_campaigns_limit(self, log):
        for i in range(1, 11):
            log.save_campaign(self._make_result(i))

        campaigns = log.get_recent_campaigns(n=3)
        assert len(campaigns) == 3
        # Should be ordered DESC by round_id
        assert campaigns[0]["round_id"] == 10
        assert campaigns[1]["round_id"] == 9
        assert campaigns[2]["round_id"] == 8

    def test_campaign_ctr_stored_correctly(self, log):
        result = self._make_result(1, impressions=200, clicks=50)
        log.save_campaign(result)

        campaigns = log.get_recent_campaigns(1)
        assert campaigns[0]["ctr"] == pytest.approx(0.25)

    def test_campaign_engagement_rate_stored(self, log):
        result = self._make_result(1, impressions=100, likes=10, shares=5)
        log.save_campaign(result)

        campaigns = log.get_recent_campaigns(1)
        assert campaigns[0]["engagement_rate"] == pytest.approx(0.15)

    def test_campaign_replace_on_duplicate_round_id(self, log):
        result1 = self._make_result(1, clicks=10)
        log.save_campaign(result1)

        result2 = self._make_result(1, clicks=99)
        log.save_campaign(result2)

        campaigns = log.get_recent_campaigns(10)
        assert len(campaigns) == 1
        assert campaigns[0]["clicks"] == 99

    # -- save_reflection & get_recent_reflections --

    def test_save_and_retrieve_reflection(self, log):
        result = self._make_result(1)
        log.save_campaign(result)

        reflection = self._make_reflection(1, result)
        log.save_reflection(reflection)

        reflections = log.get_recent_reflections(n=5)
        assert len(reflections) == 1
        assert reflections[0]["round_id"] == 1
        assert reflections[0]["key_insight"] == "精准匹配是关键"
        assert reflections[0]["what_worked"] == ["兴趣匹配准确"]
        assert reflections[0]["what_failed"] == ["低活跃用户未覆盖"]
        assert reflections[0]["root_causes"] == ["用户画像不完整"]
        assert reflections[0]["improvement_suggestions"] == ["引入更多用户特征"]

    def test_reflections_json_lists_deserialized(self, log):
        result = self._make_result(1)
        log.save_campaign(result)

        reflection = self._make_reflection(1, result)
        log.save_reflection(reflection)

        reflections = log.get_recent_reflections(1)
        r = reflections[0]
        assert isinstance(r["what_worked"], list)
        assert isinstance(r["what_failed"], list)
        assert isinstance(r["root_causes"], list)
        assert isinstance(r["improvement_suggestions"], list)

    def test_get_recent_reflections_limit(self, log):
        for i in range(1, 6):
            result = self._make_result(i)
            log.save_campaign(result)
            log.save_reflection(self._make_reflection(i, result))

        reflections = log.get_recent_reflections(n=2)
        assert len(reflections) == 2
        assert reflections[0]["round_id"] == 5
        assert reflections[1]["round_id"] == 4

    # -- get_evolution_curve --

    def test_evolution_curve_empty(self, log):
        curve = log.get_evolution_curve()
        assert curve == []

    def test_evolution_curve_ordered_by_round_id(self, log):
        for i in range(1, 4):
            log.save_campaign(self._make_result(i, impressions=100, clicks=i * 10))

        curve = log.get_evolution_curve()
        assert len(curve) == 3
        assert curve[0]["round_id"] == 1
        assert curve[1]["round_id"] == 2
        assert curve[2]["round_id"] == 3

    def test_evolution_curve_has_metrics(self, log):
        log.save_campaign(self._make_result(1, impressions=100, clicks=25, likes=10, shares=5))

        curve = log.get_evolution_curve()
        assert len(curve) == 1
        entry = curve[0]
        assert "ctr" in entry
        assert "completion_rate" in entry
        assert "engagement_rate" in entry
        assert "strategy_used" in entry
        assert entry["ctr"] == pytest.approx(0.25)
        assert entry["engagement_rate"] == pytest.approx(0.15)


# ============================================================
# StrategyLibrary Tests
# ============================================================


class TestStrategyLibrary:
    @pytest.fixture()
    def lib(self, tmp_path):
        """Create a StrategyLibrary backed by a temporary directory."""
        return StrategyLibrary(strategy_dir=tmp_path / "strategies")

    def _make_strategy(
        self,
        sid="s001",
        name="测试策略",
        scenario="宠物内容推荐",
        stype=StrategyType.NEW,
    ):
        return Strategy(
            strategy_id=sid,
            name=name,
            strategy_type=stype,
            applicable_scenario=scenario,
            target_audience="年轻宠物爱好者",
            content_direction="宠物日常视频",
            execution_steps=["分析用户兴趣", "筛选宠物内容", "排序推荐"],
            expected_effect="CTR提升10%",
        )

    # -- save & get --

    def test_save_and_get_strategy(self, lib):
        strategy = self._make_strategy()
        lib.save(strategy)

        loaded = lib.get("s001")
        assert loaded is not None
        assert loaded.strategy_id == "s001"
        assert loaded.name == "测试策略"
        assert loaded.strategy_type == StrategyType.NEW
        assert loaded.applicable_scenario == "宠物内容推荐"
        assert loaded.target_audience == "年轻宠物爱好者"
        assert loaded.content_direction == "宠物日常视频"
        assert loaded.expected_effect == "CTR提升10%"

    def test_get_preserves_execution_steps(self, lib):
        strategy = self._make_strategy()
        lib.save(strategy)

        loaded = lib.get("s001")
        assert len(loaded.execution_steps) == 3
        assert "分析用户兴趣" in loaded.execution_steps[0]

    def test_get_nonexistent_returns_none(self, lib):
        assert lib.get("nonexistent") is None

    def test_save_creates_markdown_file(self, lib):
        strategy = self._make_strategy()
        lib.save(strategy)

        md_path = lib.strategy_dir / "s001.md"
        assert md_path.exists()
        content = md_path.read_text(encoding="utf-8")
        assert "测试策略" in content
        assert "宠物内容推荐" in content

    def test_save_updates_existing_strategy(self, lib):
        s1 = self._make_strategy(name="版本1")
        lib.save(s1)

        s2 = self._make_strategy(name="版本2")
        lib.save(s2)

        loaded = lib.get("s001")
        assert loaded.name == "版本2"
        assert lib.count() == 1

    # -- list_all --

    def test_list_all_empty(self, lib):
        assert lib.list_all() == []

    def test_list_all_returns_index(self, lib):
        lib.save(self._make_strategy("s001", "策略A"))
        lib.save(self._make_strategy("s002", "策略B", scenario="美食推荐"))

        index = lib.list_all()
        assert len(index) == 2
        ids = {entry["strategy_id"] for entry in index}
        assert ids == {"s001", "s002"}

    def test_list_all_has_required_fields(self, lib):
        lib.save(self._make_strategy())

        entries = lib.list_all()
        entry = entries[0]
        assert "strategy_id" in entry
        assert "name" in entry
        assert "strategy_type" in entry
        assert "applicable_scenario" in entry
        assert "version" in entry

    # -- search --

    def test_search_by_scenario_keyword(self, lib):
        lib.save(self._make_strategy("s001", "宠物策略", "宠物内容推荐"))
        lib.save(self._make_strategy("s002", "美食策略", "美食内容推荐"))
        lib.save(self._make_strategy("s003", "科技策略", "科技内容推荐"))

        results = lib.search("宠物")
        assert len(results) == 1
        assert results[0].strategy_id == "s001"

    def test_search_case_insensitive(self, lib):
        lib.save(self._make_strategy("s001", "Pet Strategy", "pet content recommendation"))

        results = lib.search("PET")
        assert len(results) == 1

    def test_search_no_match(self, lib):
        lib.save(self._make_strategy("s001", "宠物策略", "宠物内容推荐"))

        results = lib.search("财经")
        assert results == []

    def test_search_multiple_matches(self, lib):
        lib.save(self._make_strategy("s001", "宠物策略A", "宠物狗推荐"))
        lib.save(self._make_strategy("s002", "宠物策略B", "宠物猫推荐"))

        results = lib.search("宠物")
        assert len(results) == 2

    # -- count --

    def test_count_empty(self, lib):
        assert lib.count() == 0

    def test_count_after_saves(self, lib):
        lib.save(self._make_strategy("s001"))
        lib.save(self._make_strategy("s002"))
        lib.save(self._make_strategy("s003"))
        assert lib.count() == 3

    def test_count_no_duplicate_after_overwrite(self, lib):
        lib.save(self._make_strategy("s001", "v1"))
        lib.save(self._make_strategy("s001", "v2"))
        assert lib.count() == 1

    # -- strategy with parent --

    def test_save_and_get_strategy_with_parent(self, lib):
        parent = self._make_strategy("s001", "原始策略")
        lib.save(parent)

        child = Strategy(
            strategy_id="s002",
            name="优化策略",
            strategy_type=StrategyType.REFINE,
            applicable_scenario="宠物内容推荐",
            target_audience="年轻女性",
            content_direction="萌宠视频",
            execution_steps=["步骤1"],
            expected_effect="CTR提升15%",
            parent_id="s001",
            version=2,
        )
        lib.save(child)

        loaded = lib.get("s002")
        assert loaded.parent_id == "s001"
        assert loaded.version == 2
        assert loaded.strategy_type == StrategyType.REFINE
