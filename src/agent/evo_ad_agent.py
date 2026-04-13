"""EvoAdAgent — the main evolution loop orchestrated with LangGraph.

The outer evolution cycle is also a LangGraph StateGraph:
  Execute → Reflect → Distill → (loop back or end)

This gives us two levels of LangGraph usage:
1. Inner graph (executor): ReAct tool-calling for recommendation decisions
2. Outer graph (this file): Evolution cycle orchestration

Both are genuine LangGraph StateGraph implementations with state management,
conditional edges, and proper graph architecture.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, StateGraph

from src.agent.distiller import StrategyDistiller
from src.agent.executor import AdExecutor
from src.agent.reflector import AdReflector
from src.config import ProjectConfig
from src.llm_factory import create_llm
from src.memory.campaign_log import CampaignLog
from src.memory.strategy_lib import StrategyLibrary
from src.memory.user_profile import UserProfileStore
from src.models import CampaignResult, ContentItem, Strategy, UserProfile
from src.simulation.ad_environment import AdEnvironment


# --- Evolution State ---

class EvolutionState(TypedDict):
    """State for the outer evolution graph."""

    users: list  # UserProfile objects
    contents: list  # ContentItem objects
    scenario: str
    n_users: int | None
    campaign_result: dict | None  # serialized CampaignResult
    reflection: dict | None  # serialized AdReflection
    new_strategy_name: str | None
    round_summary: dict | None


class EvoAdAgent:
    """Self-evolving recommendation agent with dual-level LangGraph architecture.

    Architecture:
    - Outer LangGraph: Evolution cycle (Execute → Reflect → Distill)
    - Inner LangGraph: ReAct tool-calling agent (in executor)
    """

    def __init__(self, config: ProjectConfig | None = None):
        self.config = (config or ProjectConfig()).resolve_all()

        # Create LLMs
        self._executor_llm = create_llm(self.config.executor_llm)
        self._reflector_llm = create_llm(self.config.reflector_llm)
        self._distiller_llm = create_llm(self.config.distiller_llm)
        self._simulator_llm = create_llm(self.config.simulator_llm)

        # Memory — all four layers
        self.campaign_log = CampaignLog(self.config.db_path)
        self.strategy_lib = StrategyLibrary(
            self.config.strategy_dir,
            emb_config=self.config.emb_config,
        )
        # L2 user profile store: built lazily in setup_environment() once we
        # have the concrete users. Requires emb_config; None disables it.
        self.user_profile_store: UserProfileStore | None = (
            UserProfileStore(emb_config=self.config.emb_config)
            if self.config.emb_config is not None
            else None
        )

        # Modules
        self.executor = AdExecutor(self._executor_llm, self.strategy_lib)
        self.reflector = AdReflector(self._reflector_llm)
        self.distiller = StrategyDistiller(self._distiller_llm)

        # Environment
        self.env: AdEnvironment | None = None

        # Build outer evolution graph
        self.evolution_graph = self._build_evolution_graph()

    def setup_environment(self, users: list[UserProfile], contents: list[ContentItem]):
        """Initialize the simulation environment and build the L2 user profile index."""
        self.env = AdEnvironment.create(users, contents, self._simulator_llm)
        if self.user_profile_store is not None and users:
            self.user_profile_store.build(users)

    def _build_evolution_graph(self) -> StateGraph:
        """Build the outer LangGraph for the evolution cycle."""

        # --- Node: Execute ---
        def execute_node(state: EvolutionState) -> dict:
            """Run the inner ReAct agent to make recommendations, then simulate."""
            users = state["users"]
            contents = state["contents"]
            scenario = state.get("scenario", "")
            n_users = state.get("n_users")

            round_users = self.env.get_users(n_users) if n_users else users
            round_contents = self.env.get_contents()

            # Inner LangGraph: ReAct agent makes decisions
            actions = self.executor.execute(round_users, round_contents, scenario)

            # Simulate user feedback. Tag this campaign with whichever strategies
            # the executor actually retrieved, not just the keyword match.
            applied = self.executor.last_applied_strategies
            strategy_name = (
                "+".join(applied[:2]) if applied else self._current_strategy_name(scenario)
            )
            result = self.env.step(actions, strategy_name)

            # Persist
            self.campaign_log.save_campaign(result)

            return {
                "campaign_result": {
                    "round_id": result.round_id,
                    "total_impressions": result.total_impressions,
                    "clicks": result.clicks,
                    "completes": result.completes,
                    "likes": result.likes,
                    "shares": result.shares,
                    "ctr": result.ctr,
                    "completion_rate": result.completion_rate,
                    "engagement_rate": result.engagement_rate,
                    "strategy_used": result.strategy_used,
                    "trajectory": result.trajectory,
                },
            }

        # --- Node: Reflect ---
        def reflect_node(state: EvolutionState) -> dict:
            """Analyze campaign results with Reflexion."""
            cr = state["campaign_result"]
            result = CampaignResult(
                round_id=cr["round_id"],
                total_impressions=cr["total_impressions"],
                clicks=cr["clicks"],
                completes=cr["completes"],
                likes=cr["likes"],
                shares=cr["shares"],
                strategy_used=cr.get("strategy_used"),
                trajectory=cr.get("trajectory", []),
            )

            history = self.campaign_log.get_evolution_curve()
            reflection = self.reflector.reflect(result, history)
            self.campaign_log.save_reflection(reflection)

            return {
                "reflection": {
                    "round_id": reflection.round_id,
                    "what_worked": reflection.what_worked,
                    "what_failed": reflection.what_failed,
                    "root_causes": reflection.root_causes,
                    "improvement_suggestions": reflection.improvement_suggestions,
                    "key_insight": reflection.key_insight,
                },
            }

        # --- Node: Distill ---
        def distill_node(state: EvolutionState) -> dict:
            """Try to extract a reusable strategy from the reflection."""
            cr = state["campaign_result"]
            refl = state["reflection"]

            result = CampaignResult(
                round_id=cr["round_id"],
                total_impressions=cr["total_impressions"],
                clicks=cr["clicks"],
                completes=cr["completes"],
                likes=cr["likes"],
                shares=cr["shares"],
            )

            from src.models import AdReflection
            reflection = AdReflection(
                round_id=refl["round_id"],
                campaign_result=result,
                what_worked=refl["what_worked"],
                what_failed=refl["what_failed"],
                root_causes=refl["root_causes"],
                improvement_suggestions=refl["improvement_suggestions"],
                key_insight=refl["key_insight"],
            )

            existing = self.strategy_lib.list_all()
            new_strategy = self.distiller.distill(reflection, existing)

            strategy_name = None
            if new_strategy:
                self.strategy_lib.save(new_strategy)
                strategy_name = new_strategy.name

            return {
                "new_strategy_name": strategy_name,
                "round_summary": {
                    "round_id": cr["round_id"],
                    "ctr": cr["ctr"],
                    "completion_rate": cr["completion_rate"],
                    "engagement_rate": cr["engagement_rate"],
                    "key_insight": refl["key_insight"],
                    "new_strategy": strategy_name,
                    "total_strategies": self.strategy_lib.count(),
                },
            }

        # --- Build graph ---
        graph = StateGraph(EvolutionState)

        graph.add_node("execute", execute_node)
        graph.add_node("reflect", reflect_node)
        graph.add_node("distill", distill_node)

        graph.set_entry_point("execute")
        graph.add_edge("execute", "reflect")
        graph.add_edge("reflect", "distill")
        graph.add_edge("distill", END)

        return graph.compile()

    def run_round(
        self,
        users: list[UserProfile] | None = None,
        contents: list[ContentItem] | None = None,
        scenario: str = "",
        n_users: int | None = None,
    ) -> dict:
        """Run one complete evolution round through the LangGraph."""
        if self.env is None:
            raise RuntimeError("Call setup_environment() first")

        initial_state: EvolutionState = {
            "users": users or self.env.users,
            "contents": contents or self.env.contents,
            "scenario": scenario,
            "n_users": n_users,
            "campaign_result": None,
            "reflection": None,
            "new_strategy_name": None,
            "round_summary": None,
        }

        final_state = self.evolution_graph.invoke(initial_state)
        return final_state.get("round_summary", {})

    def run_evolution(self, rounds: int = 10, scenario: str = "", n_users: int | None = None):
        """Run multiple rounds and print the evolution progress."""
        print(f"\n{'='*60}")
        print(f"  EvoAdAgent Evolution — {rounds} rounds")
        print(f"  Scenario: {scenario or 'general'}")
        print(f"  Initial strategies: {self.strategy_lib.count()}")
        l3_status = (
            f"{self.strategy_lib._faiss_index.ntotal} strategies"
            if self.strategy_lib.has_vector_index
            else ("configured (empty)" if self.config.emb_config else "disabled")
        )
        print(f"  L3 Strategy FAISS index: {l3_status}")
        l2_status = (
            f"{self.user_profile_store.count()} users"
            if self.user_profile_store and self.user_profile_store.count() > 0
            else "disabled"
        )
        print(f"  L2 User Profile FAISS index: {l2_status}")
        print(f"{'='*60}\n")

        for i in range(rounds):
            print(f"--- Round {i+1}/{rounds} ---")
            summary = self.run_round(scenario=scenario, n_users=n_users)

            print(f"  CTR: {summary.get('ctr', 0):.2%}")
            print(f"  Completion: {summary.get('completion_rate', 0):.2%}")
            print(f"  Engagement: {summary.get('engagement_rate', 0):.2%}")
            print(f"  Insight: {str(summary.get('key_insight', ''))[:80]}")
            applied = self.executor.last_applied_strategies
            if applied:
                print(f"  Applied strategies: {applied}")
            else:
                print(f"  Applied strategies: (none — cold start)")
            if summary.get("new_strategy"):
                print(f"  NEW STRATEGY: {summary['new_strategy']}")
            print(f"  Strategy count: {summary.get('total_strategies', 0)}")
            print()

        # Print evolution summary
        curve = self.campaign_log.get_evolution_curve()
        if len(curve) >= 2:
            first, last = curve[0], curve[-1]
            print(f"\n{'='*60}")
            print(f"  Evolution Summary")
            ctr_diff = (last['ctr'] or 0) - (first['ctr'] or 0)
            print(f"  CTR: {first['ctr']:.2%} → {last['ctr']:.2%} "
                  f"({'+'if ctr_diff > 0 else ''}{ctr_diff*100:.1f}pp)")
            print(f"  Strategies learned: {self.strategy_lib.count()}")
            print(f"{'='*60}\n")

    def _current_strategy_name(self, scenario: str) -> str | None:
        strategies = self.strategy_lib.search(scenario) if scenario else []
        return strategies[0].name if strategies else None
