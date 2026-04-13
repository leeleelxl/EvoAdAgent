"""Streamlit viz — render a benchmark_results/*.json run as interactive charts.

Usage:
    streamlit run examples/viz_evolution.py
    streamlit run examples/viz_evolution.py -- --file benchmark_results/run_XXX.json

Renders three views:
  1. Evolution curves (CTR / completion / engagement vs round)
  2. Strategy lineage forest (NEW → REFINE → MERGE)
  3. Verifier results (Cohen's d and CTR lift per strategy)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


BENCH_DIR = Path("benchmark_results")


def _list_runs() -> list[Path]:
    if not BENCH_DIR.exists():
        return []
    return sorted(BENCH_DIR.glob("run_*.json"), reverse=True)


def _load_run(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _render_header(run: dict) -> None:
    cfg = run["config"]
    summary = run["summary"]
    st.title("EvoAdAgent — Evolution Benchmark")
    st.caption(
        f"Rounds={cfg['rounds']}  Users/round={cfg['users_per_round']}  "
        f"Model={cfg['provider']}/{cfg['model']}  "
        f"Scenario={cfg['scenario'] or 'full'}  "
        f"Elapsed={run.get('elapsed_seconds', 0):.1f}s"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("First CTR", f"{summary['first_ctr']:.1%}")
    c2.metric(
        "Last CTR",
        f"{summary['last_ctr']:.1%}",
        delta=f"{summary['ctr_delta_pp']:+.1f}pp",
    )
    c3.metric("Max CTR", f"{summary['max_ctr']:.1%}")
    c4.metric("Strategies learned", summary["total_strategies"])


def _render_curve(run: dict) -> None:
    st.subheader("1) Evolution Curves")
    rounds = run["per_round"]
    if not rounds:
        st.info("No rounds recorded.")
        return

    df = pd.DataFrame(rounds)
    df_long = df.melt(
        id_vars=["round_id"],
        value_vars=["ctr", "completion_rate", "engagement_rate"],
        var_name="metric",
        value_name="rate",
    )
    fig = px.line(
        df_long,
        x="round_id",
        y="rate",
        color="metric",
        markers=True,
        title="CTR / Completion / Engagement across rounds",
    )
    fig.update_yaxes(tickformat=".0%", title="rate")
    fig.update_xaxes(title="round")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Per-round insights"):
        st.dataframe(
            df[["round_id", "ctr", "completion_rate", "engagement_rate",
                "strategy_used", "key_insight"]],
            use_container_width=True,
        )


def _render_lineage(run: dict) -> None:
    st.subheader("2) Strategy Lineage")
    strategies = run["strategies"]
    if not strategies:
        st.info("No strategies learned.")
        return

    # Build sunburst data: each strategy is a node, root is "策略库"
    ids, parents, labels, metas = ["策略库"], [""], ["策略库"], [""]
    sid_set = {s["strategy_id"] for s in strategies}
    for s in strategies:
        parent = s["parent_id"] if s["parent_id"] in sid_set else "策略库"
        ids.append(s["strategy_id"])
        parents.append(parent)
        labels.append(f"[{s['type']}] {s['name']}")
        metas.append(
            f"v{s['version']} — {s['applicable_scenario'][:40]}..."
            if s["applicable_scenario"]
            else f"v{s['version']}"
        )

    fig = go.Figure(
        go.Sunburst(
            ids=ids,
            parents=parents,
            labels=labels,
            hovertext=metas,
            branchvalues="total",
            maxdepth=3,
        )
    )
    fig.update_layout(margin=dict(t=10, l=10, r=10, b=10), height=500)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Strategy details"):
        df = pd.DataFrame(strategies)
        st.dataframe(
            df[["strategy_id", "name", "type", "parent_id", "version",
                "applicable_scenario"]],
            use_container_width=True,
        )


def _render_verification(run: dict) -> None:
    st.subheader("3) Strategy Verification (retrospective A/B)")
    verification = run["verification"]
    if not verification:
        st.info("No verification results.")
        return

    df = pd.DataFrame(verification)
    df["display_name"] = df.apply(
        lambda r: f"{r['strategy_name'][:18]}… ({r['strategy_id'][-6:]})"
        if len(r["strategy_name"]) > 18
        else f"{r['strategy_name']} ({r['strategy_id'][-6:]})",
        axis=1,
    )

    color_map = {
        "accept": "#2ca02c",
        "reject": "#d62728",
        "inconclusive_low_samples": "#7f7f7f",
        "inconclusive_no_effect": "#bcbd22",
    }

    fig = px.bar(
        df,
        x="display_name",
        y="ctr_lift_pp",
        color="verdict",
        color_discrete_map=color_map,
        hover_data=["ctr_effect_size", "n_with", "n_without", "verdict_reason"],
        title="CTR lift (percentage points) with verifier verdict",
    )
    fig.update_yaxes(title="CTR lift (pp)")
    fig.update_xaxes(title="strategy", tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

    # Effect-size chart
    fig2 = px.bar(
        df,
        x="display_name",
        y="ctr_effect_size",
        color="verdict",
        color_discrete_map=color_map,
        title="Cohen's d (effect size) per strategy",
    )
    fig2.update_yaxes(title="Cohen's d")
    fig2.update_xaxes(title="strategy", tickangle=-30)
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Verification details"):
        st.dataframe(
            df[["strategy_id", "strategy_name", "verdict", "n_with", "n_without",
                "ctr_lift_pp", "ctr_effect_size", "verdict_reason"]],
            use_container_width=True,
        )


def main() -> None:
    st.set_page_config(page_title="EvoAdAgent Evolution", layout="wide")
    runs = _list_runs()
    if not runs:
        st.error(
            "No benchmark runs found. Run `python -m examples.benchmark_run` first."
        )
        return

    st.sidebar.title("Benchmark runs")
    chosen = st.sidebar.selectbox(
        "Select a run",
        options=runs,
        format_func=lambda p: p.name,
    )

    run = _load_run(chosen)
    _render_header(run)
    st.divider()
    _render_curve(run)
    st.divider()
    _render_lineage(run)
    st.divider()
    _render_verification(run)


if __name__ == "__main__":
    main()
