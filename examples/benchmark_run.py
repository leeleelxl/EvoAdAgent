"""Benchmark run — execute N rounds end-to-end against real APIs and dump
structured metrics for viz and README consumption.

Produces `benchmark_results/run_<timestamp>.json` with:
  - config: rounds, users, scenario, provider, model
  - per_round: list of {round_id, ctr, completion_rate, engagement_rate,
    strategy_used, applied_strategies, insight, new_strategy}
  - strategies: list of {id, name, type, parent_id, scenario, ...}
  - verification: list of VerificationResult.as_dict()
  - summary: {first_ctr, last_ctr, max_ctr, ctr_delta_pp, total_strategies,
    accepted_strategies, inconclusive_strategies, rejected_strategies}

Usage:
    python -m examples.benchmark_run --rounds 20 --users 10
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from src.agent.evo_ad_agent import EvoAdAgent
from src.agent.verifier import StrategyVerifier
from src.config import LLMConfig, ProjectConfig
from src.simulation.scenarios import create_full_scenario


def _clean(config: ProjectConfig) -> None:
    if config.db_path.exists():
        config.db_path.unlink()
    if config.strategy_dir.exists():
        shutil.rmtree(config.strategy_dir)
    config.strategy_dir.mkdir(parents=True, exist_ok=True)


def _gather_per_round(agent: EvoAdAgent) -> list[dict]:
    curve = agent.campaign_log.get_evolution_curve()
    reflections = {r["round_id"]: r for r in agent.campaign_log.get_recent_reflections(n=1000)}

    out = []
    for row in curve:
        rid = row["round_id"]
        refl = reflections.get(rid, {})
        out.append({
            "round_id": rid,
            "ctr": row["ctr"],
            "completion_rate": row["completion_rate"],
            "engagement_rate": row["engagement_rate"],
            "strategy_used": row.get("strategy_used"),
            "key_insight": refl.get("key_insight", ""),
        })
    return out


def _gather_strategies(agent: EvoAdAgent) -> list[dict]:
    out = []
    for entry in agent.strategy_lib.list_all():
        s = agent.strategy_lib.get(entry["strategy_id"])
        if s is None:
            continue
        out.append({
            "strategy_id": s.strategy_id,
            "name": s.name,
            "type": s.strategy_type.value,
            "parent_id": s.parent_id,
            "version": s.version,
            "applicable_scenario": s.applicable_scenario,
            "target_audience": s.target_audience,
            "content_direction": s.content_direction,
            "execution_steps": s.execution_steps,
            "expected_effect": s.expected_effect,
            "created_at": s.created_at,
            "history_entries": len(s.historical_performance or []),
        })
    return out


def _summarize(per_round: list[dict], verification: list[dict]) -> dict:
    if not per_round:
        return {}
    ctrs = [r["ctr"] for r in per_round if r["ctr"] is not None]
    first_ctr = ctrs[0] if ctrs else 0.0
    last_ctr = ctrs[-1] if ctrs else 0.0
    max_ctr = max(ctrs) if ctrs else 0.0

    verdict_counts: dict[str, int] = {}
    for v in verification:
        verdict_counts[v["verdict"]] = verdict_counts.get(v["verdict"], 0) + 1

    return {
        "rounds": len(per_round),
        "first_ctr": round(first_ctr, 4),
        "last_ctr": round(last_ctr, 4),
        "max_ctr": round(max_ctr, 4),
        "ctr_delta_pp": round((last_ctr - first_ctr) * 100, 2),
        "mean_ctr": round(sum(ctrs) / len(ctrs), 4) if ctrs else 0.0,
        "total_strategies": len(verification),
        "verdict_counts": verdict_counts,
    }


def main():
    parser = argparse.ArgumentParser(description="EvoAdAgent benchmark run")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--users", type=int, default=10)
    parser.add_argument("--provider", type=str, default="qwen")
    parser.add_argument("--model", type=str, default="qwen-plus")
    parser.add_argument("--scenario", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="benchmark_results")
    args = parser.parse_args()

    load_dotenv()

    llm = LLMConfig(provider=args.provider, model=args.model)
    config = ProjectConfig(
        executor_llm=llm,
        reflector_llm=llm,
        distiller_llm=llm,
        simulator_llm=llm,
    )
    _clean(config)
    print(f"[clean] Wiped previous DB and strategies.\n")

    agent = EvoAdAgent(config)
    users, contents = create_full_scenario()
    agent.setup_environment(users, contents)

    print(f"Benchmark config:")
    print(f"  rounds={args.rounds} users/round={args.users} scenario={args.scenario or 'full'}")
    print(f"  model={args.provider}/{args.model}")
    print(f"  L2 user_profile_store: {agent.user_profile_store.count() if agent.user_profile_store else 0} users")
    print()

    start = time.time()
    agent.run_evolution(rounds=args.rounds, scenario=args.scenario, n_users=args.users)
    elapsed = time.time() - start

    # Verification
    verifier = StrategyVerifier()
    results = verifier.verify_all(agent.strategy_lib, agent.campaign_log)
    verification = [r.as_dict() for r in results]

    per_round = _gather_per_round(agent)
    strategies = _gather_strategies(agent)
    summary = _summarize(per_round, verification)

    # Persist results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"run_{timestamp}.json"

    payload = {
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "config": {
            "rounds": args.rounds,
            "users_per_round": args.users,
            "provider": args.provider,
            "model": args.model,
            "scenario": args.scenario,
        },
        "summary": summary,
        "per_round": per_round,
        "strategies": strategies,
        "verification": verification,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Print summary
    print(f"\n{'='*60}")
    print(f"  Benchmark Summary")
    print(f"{'='*60}")
    print(f"  Elapsed: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  CTR: first={summary['first_ctr']:.2%} → last={summary['last_ctr']:.2%}  "
          f"(Δ {summary['ctr_delta_pp']:+.2f}pp, max={summary['max_ctr']:.2%})")
    print(f"  Strategies learned: {summary['total_strategies']}")
    print(f"  Verdict distribution: {summary['verdict_counts']}")
    print(f"\n  Output: {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
