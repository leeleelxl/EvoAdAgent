"""L3: Strategy Library — stores and retrieves distilled recommendation strategies."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from src.models import Strategy


class StrategyLibrary:
    """Manages reusable recommendation strategies as searchable markdown files."""

    def __init__(self, strategy_dir: Path):
        self.strategy_dir = strategy_dir
        self.strategy_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.strategy_dir / "_index.json"

    def save(self, strategy: Strategy):
        """Save a strategy as a markdown file and update the index."""
        filepath = self.strategy_dir / f"{strategy.strategy_id}.md"
        md = self._to_markdown(strategy)
        filepath.write_text(md, encoding="utf-8")
        self._update_index(strategy)

    def get(self, strategy_id: str) -> Strategy | None:
        """Load a strategy by ID."""
        filepath = self.strategy_dir / f"{strategy_id}.md"
        if not filepath.exists():
            return None
        return self._from_markdown(filepath)

    def list_all(self) -> list[dict]:
        """Return a lightweight index of all strategies (name + scenario only)."""
        if not self._index_path.exists():
            return []
        return json.loads(self._index_path.read_text(encoding="utf-8"))

    def search(self, scenario: str) -> list[Strategy]:
        """Find strategies applicable to a given scenario (keyword match)."""
        results = []
        for entry in self.list_all():
            if scenario.lower() in entry.get("applicable_scenario", "").lower():
                strategy = self.get(entry["strategy_id"])
                if strategy:
                    results.append(strategy)
        return results

    def count(self) -> int:
        return len(self.list_all())

    def _update_index(self, strategy: Strategy):
        index = self.list_all()
        # Remove existing entry with same ID
        index = [e for e in index if e["strategy_id"] != strategy.strategy_id]
        index.append({
            "strategy_id": strategy.strategy_id,
            "name": strategy.name,
            "strategy_type": strategy.strategy_type.value,
            "applicable_scenario": strategy.applicable_scenario,
            "version": strategy.version,
        })
        self._index_path.write_text(
            json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @staticmethod
    def _to_markdown(s: Strategy) -> str:
        steps = "\n".join(f"  {i+1}. {step}" for i, step in enumerate(s.execution_steps))
        perf = ""
        if s.historical_performance:
            perf = "\n## Historical Performance\n"
            for p in s.historical_performance:
                perf += f"- Round {p.get('round_id', '?')}: CTR={p.get('ctr', '?')}, "
                perf += f"Engagement={p.get('engagement_rate', '?')}\n"
        return f"""---
strategy_id: {s.strategy_id}
name: {s.name}
type: {s.strategy_type.value}
version: {s.version}
created_at: {s.created_at}
parent_id: {s.parent_id or 'none'}
---

# {s.name}

## Applicable Scenario
{s.applicable_scenario}

## Target Audience
{s.target_audience}

## Content Direction
{s.content_direction}

## Execution Steps
{steps}

## Expected Effect
{s.expected_effect}
{perf}"""

    @staticmethod
    def _from_markdown(filepath: Path) -> Strategy:
        text = filepath.read_text(encoding="utf-8")
        # Parse frontmatter
        parts = text.split("---")
        if len(parts) < 3:
            return None
        meta = {}
        for line in parts[1].strip().split("\n"):
            if ":" in line:
                key, val = line.split(":", 1)
                meta[key.strip()] = val.strip()
        # Parse body sections
        body = "---".join(parts[2:])
        sections = {}
        current_section = None
        for line in body.split("\n"):
            if line.startswith("## "):
                current_section = line[3:].strip()
                sections[current_section] = []
            elif current_section:
                sections[current_section].append(line)

        def get_section(name: str) -> str:
            return "\n".join(sections.get(name, [])).strip()

        steps_raw = get_section("Execution Steps")
        steps = [line.strip().lstrip("0123456789. ") for line in steps_raw.split("\n") if line.strip()]

        from src.models import StrategyType
        return Strategy(
            strategy_id=meta.get("strategy_id", filepath.stem),
            name=meta.get("name", ""),
            strategy_type=StrategyType(meta.get("type", "new")),
            applicable_scenario=get_section("Applicable Scenario"),
            target_audience=get_section("Target Audience"),
            content_direction=get_section("Content Direction"),
            execution_steps=steps,
            expected_effect=get_section("Expected Effect"),
            version=int(meta.get("version", 1)),
            created_at=meta.get("created_at", ""),
            parent_id=meta.get("parent_id") if meta.get("parent_id") != "none" else None,
        )
