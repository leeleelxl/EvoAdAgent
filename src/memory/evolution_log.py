"""L4: Evolution Log — queries campaign history from an evolution perspective.

Thin wrapper over CampaignLog's SQLite. The data is already there; this module
just reshapes it for evolution-specific questions:
  - CTR / completion / engagement curves per round
  - Which strategies were active at which round
  - Strategy lineage tree (NEW → REFINE → MERGE chains)

The raw SQLite data in `campaigns` remains the single source of truth — no
duplicate storage. This is a view, not a cache.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from src.memory.campaign_log import CampaignLog
from src.memory.strategy_lib import StrategyLibrary


@dataclass
class EvolutionCurvePoint:
    round_id: int
    ctr: float
    completion_rate: float
    engagement_rate: float
    strategies_applied: list[str] = field(default_factory=list)


@dataclass
class StrategyLineageNode:
    strategy_id: str
    name: str
    type: str  # new | refine | merge
    parent_id: str | None
    version: int
    children: list["StrategyLineageNode"] = field(default_factory=list)


class EvolutionLog:
    """Evolution-perspective view over CampaignLog + StrategyLibrary."""

    def __init__(self, campaign_log: CampaignLog, strategy_lib: StrategyLibrary):
        self.campaign_log = campaign_log
        self.strategy_lib = strategy_lib

    def curve(self) -> list[EvolutionCurvePoint]:
        """Per-round CTR / completion / engagement with applied strategies parsed."""
        out: list[EvolutionCurvePoint] = []
        for row in self.campaign_log.get_evolution_curve():
            used_raw = row.get("strategy_used") or ""
            strategies_applied = [s for s in used_raw.split("+") if s]
            out.append(
                EvolutionCurvePoint(
                    round_id=row["round_id"],
                    ctr=row["ctr"] or 0.0,
                    completion_rate=row["completion_rate"] or 0.0,
                    engagement_rate=row["engagement_rate"] or 0.0,
                    strategies_applied=strategies_applied,
                )
            )
        return out

    def lineage_roots(self) -> list[StrategyLineageNode]:
        """Return the strategy forest rooted at strategies with no parent.

        Children are REFINE/MERGE strategies that point back via `parent_id`.
        """
        all_strategies = []
        for entry in self.strategy_lib.list_all():
            s = self.strategy_lib.get(entry["strategy_id"])
            if s is not None:
                all_strategies.append(s)

        nodes: dict[str, StrategyLineageNode] = {}
        for s in all_strategies:
            nodes[s.strategy_id] = StrategyLineageNode(
                strategy_id=s.strategy_id,
                name=s.name,
                type=s.strategy_type.value,
                parent_id=s.parent_id,
                version=s.version,
            )

        roots: list[StrategyLineageNode] = []
        for node in nodes.values():
            if node.parent_id and node.parent_id in nodes:
                nodes[node.parent_id].children.append(node)
            else:
                roots.append(node)
        return roots

    def ctr_delta(self) -> dict:
        """Summary: first CTR → last CTR → delta (pp) + max."""
        points = self.curve()
        if not points:
            return {"first_ctr": 0.0, "last_ctr": 0.0, "delta_pp": 0.0, "max_ctr": 0.0}
        ctrs = [p.ctr for p in points]
        return {
            "rounds": len(points),
            "first_ctr": ctrs[0],
            "last_ctr": ctrs[-1],
            "max_ctr": max(ctrs),
            "mean_ctr": sum(ctrs) / len(ctrs),
            "delta_pp": (ctrs[-1] - ctrs[0]) * 100,
        }

    def strategy_usage_counts(self) -> dict[str, int]:
        """How many rounds applied each strategy."""
        counts: dict[str, int] = {}
        for p in self.curve():
            for sid in p.strategies_applied:
                counts[sid] = counts.get(sid, 0) + 1
        return counts
