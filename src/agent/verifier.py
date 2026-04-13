"""Strategy Verifier — retrospective A/B evaluation from the campaign log.

Approach: instead of re-running costly live experiments per candidate strategy,
we mine the existing `campaigns` table. Each campaign row already records which
strategies were applied (via `strategy_used`). For each strategy S we split
rounds into two groups:

  - WITH:    S appears in campaign.strategy_used
  - WITHOUT: S does not

We report the per-metric lift (WITH minus WITHOUT) and Cohen's d effect size.
Verdict is purely descriptive with a clear inconclusive band — the code never
hides low-sample noise behind a false "accept".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from src.memory.campaign_log import CampaignLog
from src.memory.strategy_lib import StrategyLibrary


Verdict = str  # "accept" | "reject" | "inconclusive_low_samples" | "inconclusive_no_effect"


@dataclass
class VerificationResult:
    strategy_id: str
    strategy_name: str
    n_with: int
    n_without: int

    with_ctr: float
    without_ctr: float
    ctr_lift: float
    ctr_effect_size: float

    with_completion: float
    without_completion: float
    completion_lift: float

    with_engagement: float
    without_engagement: float
    engagement_lift: float

    verdict: Verdict
    verdict_reason: str = ""

    def as_dict(self) -> dict:
        return {
            "strategy_id": self.strategy_id,
            "strategy_name": self.strategy_name,
            "n_with": self.n_with,
            "n_without": self.n_without,
            "ctr_lift_pp": round(self.ctr_lift * 100, 2),
            "ctr_effect_size": round(self.ctr_effect_size, 3),
            "completion_lift_pp": round(self.completion_lift * 100, 2),
            "engagement_lift_pp": round(self.engagement_lift * 100, 2),
            "verdict": self.verdict,
            "verdict_reason": self.verdict_reason,
        }


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _cohen_d(a: list[float], b: list[float]) -> float:
    """Cohen's d standardized mean difference. 0 when either group is empty
    or both groups are constant (no information)."""
    if not a or not b:
        return 0.0
    m_a, m_b = _mean(a), _mean(b)
    if len(a) < 2 and len(b) < 2:
        # Can't estimate spread from singletons.
        return 0.0
    var_a = sum((x - m_a) ** 2 for x in a) / max(len(a) - 1, 1)
    var_b = sum((x - m_b) ** 2 for x in b) / max(len(b) - 1, 1)
    pooled_sd = math.sqrt(max((var_a + var_b) / 2.0, 0.0))
    if pooled_sd < 1e-9:
        return 0.0
    return (m_a - m_b) / pooled_sd


class StrategyVerifier:
    """Retrospective A/B tester against historical campaign data."""

    def __init__(
        self,
        min_samples_per_side: int = 2,
        min_effect_size: float = 0.3,
        min_ctr_lift: float = 0.02,  # 2 percentage points
    ):
        self.min_samples_per_side = min_samples_per_side
        self.min_effect_size = min_effect_size
        self.min_ctr_lift = min_ctr_lift

    def evaluate(
        self,
        strategy_id: str,
        strategy_name: str,
        campaign_log: CampaignLog,
    ) -> VerificationResult:
        """Compare campaigns that applied `strategy_id` against those that didn't."""
        curve = campaign_log.get_evolution_curve()
        with_group, without_group = self._split(curve, strategy_id)

        with_ctr = [c["ctr"] for c in with_group if c["ctr"] is not None]
        without_ctr = [c["ctr"] for c in without_group if c["ctr"] is not None]
        with_comp = [c["completion_rate"] for c in with_group if c["completion_rate"] is not None]
        without_comp = [c["completion_rate"] for c in without_group if c["completion_rate"] is not None]
        with_eng = [c["engagement_rate"] for c in with_group if c["engagement_rate"] is not None]
        without_eng = [c["engagement_rate"] for c in without_group if c["engagement_rate"] is not None]

        ctr_lift = _mean(with_ctr) - _mean(without_ctr)
        ctr_d = _cohen_d(with_ctr, without_ctr)

        verdict, reason = self._decide(
            n_with=len(with_group),
            n_without=len(without_group),
            ctr_lift=ctr_lift,
            ctr_d=ctr_d,
        )

        return VerificationResult(
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            n_with=len(with_group),
            n_without=len(without_group),
            with_ctr=_mean(with_ctr),
            without_ctr=_mean(without_ctr),
            ctr_lift=ctr_lift,
            ctr_effect_size=ctr_d,
            with_completion=_mean(with_comp),
            without_completion=_mean(without_comp),
            completion_lift=_mean(with_comp) - _mean(without_comp),
            with_engagement=_mean(with_eng),
            without_engagement=_mean(without_eng),
            engagement_lift=_mean(with_eng) - _mean(without_eng),
            verdict=verdict,
            verdict_reason=reason,
        )

    def verify_all(
        self,
        strategy_lib: StrategyLibrary,
        campaign_log: CampaignLog,
    ) -> list[VerificationResult]:
        """Evaluate every strategy currently in the library."""
        return [
            self.evaluate(entry["strategy_id"], entry["name"], campaign_log)
            for entry in strategy_lib.list_all()
        ]

    # --- Internals ---

    @staticmethod
    def _split(curve: list[dict], strategy_id: str) -> tuple[list[dict], list[dict]]:
        """Split campaign rows by whether they applied the given strategy."""
        with_group: list[dict] = []
        without_group: list[dict] = []
        for row in curve:
            used = row.get("strategy_used") or ""
            if strategy_id and strategy_id in used:
                with_group.append(row)
            else:
                without_group.append(row)
        return with_group, without_group

    def _decide(
        self,
        n_with: int,
        n_without: int,
        ctr_lift: float,
        ctr_d: float,
    ) -> tuple[Verdict, str]:
        if n_with < self.min_samples_per_side or n_without < self.min_samples_per_side:
            return (
                "inconclusive_low_samples",
                f"n_with={n_with}, n_without={n_without}; need ≥{self.min_samples_per_side} each",
            )
        if ctr_lift >= self.min_ctr_lift and ctr_d >= self.min_effect_size:
            return (
                "accept",
                f"CTR lift +{ctr_lift*100:.2f}pp, Cohen's d={ctr_d:.2f}",
            )
        if ctr_lift <= -self.min_ctr_lift and ctr_d <= -self.min_effect_size:
            return (
                "reject",
                f"CTR drop {ctr_lift*100:.2f}pp, Cohen's d={ctr_d:.2f}",
            )
        return (
            "inconclusive_no_effect",
            f"CTR lift {ctr_lift*100:+.2f}pp, Cohen's d={ctr_d:+.2f} below thresholds",
        )
