"""Tests for StrategyVerifier — retrospective A/B evaluation."""

from __future__ import annotations

import pytest

from src.agent.verifier import StrategyVerifier, _cohen_d
from src.memory.campaign_log import CampaignLog
from src.models import CampaignResult


def _make_campaign(round_id: int, ctr: float = 0.3, strategy_used: str | None = None) -> CampaignResult:
    impressions = 100
    clicks = int(ctr * impressions)
    return CampaignResult(
        round_id=round_id,
        total_impressions=impressions,
        clicks=clicks,
        completes=int(ctr * 50),  # completion tracks CTR loosely
        likes=int(ctr * 20),
        shares=int(ctr * 10),
        strategy_used=strategy_used,
    )


@pytest.fixture()
def log(tmp_path):
    return CampaignLog(tmp_path / "test.db")


class TestCohenD:
    def test_zero_when_empty(self):
        assert _cohen_d([], [1, 2]) == 0.0
        assert _cohen_d([1, 2], []) == 0.0

    def test_zero_when_equal_distributions(self):
        assert _cohen_d([0.1, 0.2, 0.3], [0.1, 0.2, 0.3]) == pytest.approx(0.0, abs=1e-6)

    def test_positive_when_a_greater(self):
        a = [0.5, 0.6, 0.7]
        b = [0.1, 0.2, 0.3]
        assert _cohen_d(a, b) > 0

    def test_negative_when_a_less(self):
        a = [0.1, 0.2, 0.3]
        b = [0.5, 0.6, 0.7]
        assert _cohen_d(a, b) < 0

    def test_handles_singletons_without_crash(self):
        # Both groups size 1 → no spread estimate → return 0
        assert _cohen_d([0.5], [0.3]) == 0.0


class TestSplit:
    def test_split_groups_by_strategy_id_substring(self, log):
        log.save_campaign(_make_campaign(1, ctr=0.4, strategy_used="strat_abc"))
        log.save_campaign(_make_campaign(2, ctr=0.5, strategy_used="strat_xyz+strat_abc"))
        log.save_campaign(_make_campaign(3, ctr=0.2, strategy_used="strat_other"))
        log.save_campaign(_make_campaign(4, ctr=0.1, strategy_used=None))

        v = StrategyVerifier()
        r = v.evaluate("strat_abc", "Alpha", log)
        assert r.n_with == 2  # rounds 1 and 2
        assert r.n_without == 2  # rounds 3 and 4


class TestVerdict:
    def test_low_samples_always_inconclusive(self, log):
        log.save_campaign(_make_campaign(1, ctr=0.9, strategy_used="S1"))
        log.save_campaign(_make_campaign(2, ctr=0.1, strategy_used=None))

        v = StrategyVerifier(min_samples_per_side=2)
        r = v.evaluate("S1", "Strategy1", log)
        assert r.verdict == "inconclusive_low_samples"
        assert "n_with=1" in r.verdict_reason

    def test_accept_when_large_positive_lift(self, log):
        # With strategy: CTR ≈ 0.45-0.55 (mean 0.5); without: 0.08-0.12 (mean 0.1)
        for i, ctr in enumerate([0.45, 0.50, 0.55]):
            log.save_campaign(_make_campaign(i, ctr=ctr, strategy_used="S1"))
        for i, ctr in enumerate([0.08, 0.10, 0.12], start=3):
            log.save_campaign(_make_campaign(i, ctr=ctr, strategy_used="other"))

        v = StrategyVerifier(min_samples_per_side=2, min_effect_size=0.3, min_ctr_lift=0.02)
        r = v.evaluate("S1", "Strategy1", log)
        assert r.verdict == "accept", f"expected accept, got {r.verdict}: {r.verdict_reason}"
        assert r.ctr_lift == pytest.approx(0.4, abs=0.02)

    def test_reject_when_strategy_hurts(self, log):
        for i, ctr in enumerate([0.08, 0.10, 0.12]):
            log.save_campaign(_make_campaign(i, ctr=ctr, strategy_used="Sbad"))
        for i, ctr in enumerate([0.45, 0.50, 0.55], start=3):
            log.save_campaign(_make_campaign(i, ctr=ctr, strategy_used=None))

        v = StrategyVerifier(min_samples_per_side=2)
        r = v.evaluate("Sbad", "BadStrategy", log)
        assert r.verdict == "reject", f"expected reject, got {r.verdict}: {r.verdict_reason}"
        assert r.ctr_lift < 0

    def test_inconclusive_when_effect_below_threshold(self, log):
        # Tiny lift, below threshold
        for i in range(3):
            log.save_campaign(_make_campaign(i, ctr=0.30, strategy_used="S1"))
        for i in range(3, 6):
            log.save_campaign(_make_campaign(i, ctr=0.29, strategy_used=None))

        v = StrategyVerifier(min_samples_per_side=2, min_effect_size=0.3, min_ctr_lift=0.02)
        r = v.evaluate("S1", "Strategy1", log)
        assert r.verdict == "inconclusive_no_effect"


class TestEvaluateFields:
    def test_all_metrics_populated(self, log):
        for i in range(2):
            log.save_campaign(_make_campaign(i, ctr=0.5, strategy_used="S1"))
        for i in range(2, 4):
            log.save_campaign(_make_campaign(i, ctr=0.1, strategy_used=None))

        v = StrategyVerifier()
        r = v.evaluate("S1", "Strategy1", log)
        assert r.strategy_id == "S1"
        assert r.strategy_name == "Strategy1"
        assert r.with_ctr > r.without_ctr
        assert r.with_completion >= 0
        assert r.with_engagement >= 0
        # Serialization
        d = r.as_dict()
        assert d["ctr_lift_pp"] == pytest.approx(40.0, abs=0.1)
        assert "verdict" in d


class TestVerifyAll:
    def test_verify_all_covers_every_strategy(self, log, tmp_path):
        from src.memory.strategy_lib import StrategyLibrary
        from src.models import Strategy, StrategyType

        lib = StrategyLibrary(strategy_dir=tmp_path / "strategies")
        for sid in ("A", "B", "C"):
            lib.save(
                Strategy(
                    strategy_id=sid,
                    name=f"strat_{sid}",
                    strategy_type=StrategyType.NEW,
                    applicable_scenario="test",
                    target_audience="test",
                    content_direction="test",
                    execution_steps=["step"],
                    expected_effect="test",
                )
            )

        # No campaigns referencing them → all low-sample
        v = StrategyVerifier()
        results = v.verify_all(lib, log)
        assert len(results) == 3
        assert {r.strategy_id for r in results} == {"A", "B", "C"}
        assert all(r.verdict == "inconclusive_low_samples" for r in results)
