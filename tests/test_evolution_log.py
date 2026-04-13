"""Tests for EvolutionLog — the L4 view layer over campaigns + strategies."""

from __future__ import annotations

import pytest

from src.memory.campaign_log import CampaignLog
from src.memory.evolution_log import EvolutionLog, StrategyLineageNode
from src.memory.strategy_lib import StrategyLibrary
from src.models import CampaignResult, Strategy, StrategyType


def _make_campaign(round_id: int, ctr: float, strategy_used: str | None = None) -> CampaignResult:
    return CampaignResult(
        round_id=round_id,
        total_impressions=100,
        clicks=int(ctr * 100),
        completes=int(ctr * 50),
        likes=int(ctr * 20),
        shares=int(ctr * 10),
        strategy_used=strategy_used,
    )


@pytest.fixture()
def setup(tmp_path):
    log = CampaignLog(tmp_path / "test.db")
    lib = StrategyLibrary(strategy_dir=tmp_path / "strategies")
    return log, lib, EvolutionLog(log, lib)


class TestCurve:
    def test_empty(self, setup):
        _, _, elog = setup
        assert elog.curve() == []

    def test_populated(self, setup):
        log, _, elog = setup
        log.save_campaign(_make_campaign(1, 0.2, "S1"))
        log.save_campaign(_make_campaign(2, 0.3, "S1+S2"))
        points = elog.curve()
        assert len(points) == 2
        assert points[0].round_id == 1
        assert points[0].ctr == pytest.approx(0.2)
        assert points[1].strategies_applied == ["S1", "S2"]

    def test_parses_plus_separated_strategies(self, setup):
        log, _, elog = setup
        log.save_campaign(_make_campaign(1, 0.3, "A+B+C"))
        points = elog.curve()
        assert points[0].strategies_applied == ["A", "B", "C"]

    def test_empty_strategy_used(self, setup):
        log, _, elog = setup
        log.save_campaign(_make_campaign(1, 0.1, None))
        points = elog.curve()
        assert points[0].strategies_applied == []


class TestLineage:
    def _make(self, sid, ptype=StrategyType.NEW, parent=None, version=1):
        return Strategy(
            strategy_id=sid,
            name=f"strat_{sid}",
            strategy_type=ptype,
            applicable_scenario="t",
            target_audience="t",
            content_direction="t",
            execution_steps=["s"],
            expected_effect="t",
            parent_id=parent,
            version=version,
        )

    def test_empty_lineage(self, setup):
        _, _, elog = setup
        assert elog.lineage_roots() == []

    def test_flat_forest(self, setup):
        _, lib, elog = setup
        lib.save(self._make("A"))
        lib.save(self._make("B"))
        roots = elog.lineage_roots()
        assert len(roots) == 2
        assert all(not r.children for r in roots)

    def test_parent_child_chain(self, setup):
        _, lib, elog = setup
        lib.save(self._make("root"))
        lib.save(self._make("child", ptype=StrategyType.REFINE, parent="root", version=2))
        roots = elog.lineage_roots()
        assert len(roots) == 1
        assert roots[0].strategy_id == "root"
        assert len(roots[0].children) == 1
        assert roots[0].children[0].strategy_id == "child"
        assert roots[0].children[0].type == "refine"

    def test_orphan_becomes_root(self, setup):
        _, lib, elog = setup
        # Parent_id points to nonexistent strategy → orphan is treated as root
        lib.save(self._make("orphan", ptype=StrategyType.REFINE, parent="ghost"))
        roots = elog.lineage_roots()
        assert len(roots) == 1
        assert roots[0].strategy_id == "orphan"


class TestSummaries:
    def test_ctr_delta_on_empty(self, setup):
        _, _, elog = setup
        d = elog.ctr_delta()
        assert d["delta_pp"] == 0.0

    def test_ctr_delta_positive(self, setup):
        log, _, elog = setup
        log.save_campaign(_make_campaign(1, 0.1))
        log.save_campaign(_make_campaign(2, 0.4))
        d = elog.ctr_delta()
        assert d["first_ctr"] == pytest.approx(0.1)
        assert d["last_ctr"] == pytest.approx(0.4)
        assert d["delta_pp"] == pytest.approx(30.0)
        assert d["max_ctr"] == pytest.approx(0.4)
        assert d["rounds"] == 2

    def test_strategy_usage_counts(self, setup):
        log, _, elog = setup
        log.save_campaign(_make_campaign(1, 0.2, "A"))
        log.save_campaign(_make_campaign(2, 0.3, "A+B"))
        log.save_campaign(_make_campaign(3, 0.3, "B"))
        counts = elog.strategy_usage_counts()
        assert counts == {"A": 2, "B": 2}
