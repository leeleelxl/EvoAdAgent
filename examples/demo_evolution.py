"""Demo: Watch EvoAdAgent learn and evolve over multiple rounds.

Usage:
    python -m examples.demo_evolution
    python -m examples.demo_evolution --rounds 5 --scenario 宠物
    python -m examples.demo_evolution --provider qwen --rounds 10
"""

from __future__ import annotations

import argparse

from src.agent.evo_ad_agent import EvoAdAgent
from src.config import LLMConfig, ProjectConfig
from src.simulation.scenarios import (
    create_full_scenario,
    create_food_scenario,
    create_pet_scenario,
)


def main():
    parser = argparse.ArgumentParser(description="EvoAdAgent Evolution Demo")
    parser.add_argument("--rounds", type=int, default=5, help="Number of evolution rounds")
    parser.add_argument("--scenario", type=str, default="", help="Scenario filter (宠物/美食/empty=all)")
    parser.add_argument("--provider", type=str, default="openai", help="LLM provider (openai/qwen/deepseek)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model name")
    parser.add_argument("--users", type=int, default=10, help="Number of users per round")
    args = parser.parse_args()

    # Configure LLM
    llm_config = LLMConfig(provider=args.provider, model=args.model)
    config = ProjectConfig(
        executor_llm=llm_config,
        reflector_llm=llm_config,
        distiller_llm=LLMConfig(provider=args.provider, model=args.model),
        simulator_llm=llm_config,
    )

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

    print("Done! Check the strategies/ directory for learned strategies.")
    print(f"Database: {config.db_path}")


if __name__ == "__main__":
    main()
