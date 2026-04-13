"""Render the same three charts as the Streamlit app but save to PNG so they
can be committed to the repo and referenced from README.

Usage:
    python -m examples.render_viz_pngs
    python -m examples.render_viz_pngs --file benchmark_results/run_XXX.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _curve_fig(run: dict):
    df = pd.DataFrame(run["per_round"])
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
        title="Evolution Curves — 20 rounds (Qwen-plus, full scenario)",
    )
    fig.update_yaxes(tickformat=".0%", title="rate")
    fig.update_xaxes(title="round", dtick=1)
    fig.update_layout(width=1000, height=500, template="plotly_white")
    return fig


def _lineage_fig(run: dict):
    strategies = run["strategies"]
    ids, parents, labels, metas = ["策略库"], [""], ["策略库"], [""]
    sid_set = {s["strategy_id"] for s in strategies}
    for s in strategies:
        parent = s["parent_id"] if s["parent_id"] in sid_set else "策略库"
        ids.append(s["strategy_id"])
        parents.append(parent)
        labels.append(f"[{s['type']}] {s['name'][:14]}…")
        metas.append(s["applicable_scenario"][:40])

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
    fig.update_layout(
        title="Strategy Lineage — 20 learned strategies",
        width=800,
        height=700,
        template="plotly_white",
        margin=dict(t=60, l=10, r=10, b=10),
    )
    return fig


def _verdict_fig(run: dict):
    df = pd.DataFrame(run["verification"])
    df = df.sort_values("ctr_lift_pp", ascending=False)
    df["display_name"] = df.apply(
        lambda r: f"{r['strategy_name'][:16]}…\n({r['strategy_id'][-6:]})", axis=1
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
        title="Strategy Verification — CTR lift (pp) per strategy (retrospective A/B)",
        hover_data=["ctr_effect_size", "n_with", "n_without"],
    )
    fig.update_yaxes(title="CTR lift (percentage points)")
    fig.update_xaxes(title="strategy", tickangle=-30)
    fig.update_layout(width=1400, height=600, template="plotly_white")
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None,
                        help="Benchmark JSON path (default: latest)")
    parser.add_argument("--out-dir", type=str, default="docs/images")
    args = parser.parse_args()

    bench_dir = Path("benchmark_results")
    if args.file:
        run_path = Path(args.file)
    else:
        runs = sorted(bench_dir.glob("run_*.json"), reverse=True)
        if not runs:
            print("No benchmark runs found.")
            return
        run_path = runs[0]

    print(f"Rendering from {run_path}")
    run = _load(run_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, fig in [
        ("evolution_curve", _curve_fig(run)),
        ("strategy_lineage", _lineage_fig(run)),
        ("verification_verdicts", _verdict_fig(run)),
    ]:
        out_path = out_dir / f"{name}.png"
        fig.write_image(out_path, scale=2)
        print(f"  wrote {out_path}  ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
