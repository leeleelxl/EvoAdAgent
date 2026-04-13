"""Demo: Watch EvoAdAgent learn and evolve over multiple rounds.

Usage:
    python -m examples.demo_evolution
    python -m examples.demo_evolution --rounds 5 --scenario 宠物
    python -m examples.demo_evolution --provider qwen --rounds 10
"""

from __future__ import annotations

import argparse
import shutil

from src.agent.evo_ad_agent import EvoAdAgent
from src.agent.verifier import StrategyVerifier
from src.config import LLMConfig, ProjectConfig
from src.simulation.scenarios import (
    create_full_scenario,
    create_food_scenario,
    create_pet_scenario,
)


def _clean_session(config: ProjectConfig) -> None:
    """Wipe DB + strategy dir + FAISS artifacts so the demo runs from scratch.

    Evolution summary aggregates over the entire campaign_log, so without this
    repeated demo runs bleed round counts and CTR deltas across sessions.
    """
    if config.db_path.exists():
        config.db_path.unlink()
    if config.strategy_dir.exists():
        shutil.rmtree(config.strategy_dir)
    config.strategy_dir.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="EvoAdAgent Evolution Demo")
    parser.add_argument("--rounds", type=int, default=5, help="Number of evolution rounds")
    parser.add_argument("--scenario", type=str, default="", help="Scenario filter (宠物/美食/empty=all)")
    parser.add_argument("--provider", type=str, default="openai", help="LLM provider (openai/qwen/deepseek)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model name")
    parser.add_argument("--users", type=int, default=10, help="Number of users per round")
    parser.add_argument("--clean", action="store_true",
                        help="Wipe DB and strategy dir before running (recommended for fair evolution demos)")
    parser.add_argument("--verify", action="store_true",
                        help="After all rounds, run StrategyVerifier (retrospective A/B) on the learned strategies")
    args = parser.parse_args()

    # Configure LLM
    llm_config = LLMConfig(provider=args.provider, model=args.model)
    config = ProjectConfig(
        executor_llm=llm_config,
        reflector_llm=llm_config,
        distiller_llm=LLMConfig(provider=args.provider, model=args.model),
        simulator_llm=llm_config,
    )

    if args.clean:
        _clean_session(config)
        print("[clean] wiped previous DB and strategies.\n")

    # Create agent
    agent = EvoAdAgent(config)

    # Load scenario
    if args.scenario == "宠物":
        users, contents = create_pet_scenario()
    elif args.scenario == "美食":
        users, contents = create_food_scenario()
    else:
        users, contents = create_full_scenario()

    # Setup environment
    agent.setup_environment(users, contents)

    # Run evolution
    agent.run_evolution(rounds=args.rounds, scenario=args.scenario, n_users=args.users)

    if args.verify:
        _print_verification(agent)

    print("Done! Check the strategies/ directory for learned strategies.")
    print(f"Database: {config.db_path}")


def _print_verification(agent: EvoAdAgent) -> None:
    verifier = StrategyVerifier()
    results = verifier.verify_all(agent.strategy_lib, agent.campaign_log)
    if not results:
        return
    print("\n" + "=" * 60)
    print("  Strategy Verification (retrospective A/B on campaign_log)")
    print("=" * 60)
    for r in results:
        print(f"\n  [{r.verdict}] {r.strategy_name} ({r.strategy_id})")
        print(f"    n_with={r.n_with}  n_without={r.n_without}")
        print(f"    CTR: {r.with_ctr:.2%} vs {r.without_ctr:.2%}  "
              f"(lift {r.ctr_lift*100:+.2f}pp, d={r.ctr_effect_size:+.2f})")
        print(f"    Completion: lift {r.completion_lift*100:+.2f}pp")
        print(f"    Engagement: lift {r.engagement_lift*100:+.2f}pp")
        print(f"    Reason: {r.verdict_reason}")


if __name__ == "__main__":
    main()
