"""Memory layers for EvoAdAgent.

L1 campaign_log: SQLite persistence of campaign trajectories.
L2 user_profile: FAISS vector index over user personas.
L3 strategy_lib: Distilled recommendation strategies.
L4 evolution_log: TODO — evolution history + visualization.
"""

from src.memory.campaign_log import CampaignLog
from src.memory.evolution_log import (
    EvolutionCurvePoint,
    EvolutionLog,
    StrategyLineageNode,
)
from src.memory.strategy_lib import StrategyLibrary
from src.memory.user_profile import UserProfileStore, user_to_persona_text

__all__ = [
    "CampaignLog",
    "EvolutionLog",
    "EvolutionCurvePoint",
    "StrategyLineageNode",
    "StrategyLibrary",
    "UserProfileStore",
    "user_to_persona_text",
]
