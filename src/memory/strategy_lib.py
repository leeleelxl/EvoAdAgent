"""L3: Strategy Library — stores and retrieves distilled recommendation strategies.

Dual retrieval:
  - keyword search() — exact/substring match on applicable_scenario
  - semantic_search() — FAISS vector index over (name + scenario + audience + direction)
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Align with user_profile: set before any downstream code imports faiss.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402

from src.config import LLMConfig
from src.models import Strategy


def strategy_to_signature(s: Strategy) -> str:
    """Compact semantic signature for embedding.

    Intentionally excludes execution_steps — those are too detailed and add
    noise to similarity. The signature focuses on *what* and *who*, not *how*.
    """
    return (
        f"{s.name}。适用场景：{s.applicable_scenario}。"
        f"目标人群：{s.target_audience}。内容方向：{s.content_direction}"
    )


class StrategyLibrary:
    """Manages reusable recommendation strategies as searchable markdown files.

    If ``emb_config`` is provided, strategies are also indexed in FAISS for
    semantic retrieval via ``semantic_search()``. Without it, only keyword
    search is available — existing callers keep working unchanged.
    """

    def __init__(
        self,
        strategy_dir: Path,
        emb_config: LLMConfig | None = None,
    ):
        self.strategy_dir = strategy_dir
        self.strategy_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.strategy_dir / "_index.json"
        self._faiss_path = self.strategy_dir / "_vectors.faiss"
        self._vec_meta_path = self.strategy_dir / "_vectors_meta.json"

        self.emb_config = emb_config
        self._embedder: Any | None = None
        self._faiss_index: Any | None = None
        self._faiss_dim: int | None = None
        self._faiss_ids: list[str] = []
        if self.emb_config is not None:
            self._load_faiss()

    @property
    def embedder(self):
        if self._embedder is None and self.emb_config is not None:
            from src.llm_factory import create_embeddings
            self._embedder = create_embeddings(self.emb_config)
        return self._embedder

    @property
    def has_vector_index(self) -> bool:
        return self._faiss_index is not None and self._faiss_index.ntotal > 0

    def save(self, strategy: Strategy):
        """Save a strategy as a markdown file and update both indices."""
        filepath = self.strategy_dir / f"{strategy.strategy_id}.md"
        md = self._to_markdown(strategy)
        filepath.write_text(md, encoding="utf-8")
        self._update_index(strategy)
        if self.emb_config is not None:
            self._upsert_vector(strategy)

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

    def semantic_search(self, query: str, k: int = 5) -> list[tuple[Strategy, float]]:
        """Semantic search over (name + scenario + audience + direction).

        Returns [(strategy, l2_distance), ...] sorted by similarity (lower = closer).
        Returns [] if no vector index is configured or empty.
        """
        if not self.has_vector_index:
            return []
        vec = np.array([self.embedder.embed_query(query)], dtype="float32")
        k = min(k, self._faiss_index.ntotal)
        distances, indices = self._faiss_index.search(vec, k)
        out: list[tuple[Strategy, float]] = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self._faiss_ids):
                continue
            sid = self._faiss_ids[idx]
            s = self.get(sid)
            if s:
                out.append((s, float(dist)))
        return out

    def rebuild_vector_index(self) -> int:
        """Re-embed every strategy from disk and rebuild FAISS from scratch."""
        if self.emb_config is None:
            raise RuntimeError("No emb_config — cannot build vector index")

        import faiss

        entries = self.list_all()
        strategies = [self.get(e["strategy_id"]) for e in entries]
        strategies = [s for s in strategies if s is not None]
        if not strategies:
            self._faiss_index = None
            self._faiss_dim = None
            self._faiss_ids = []
            return 0

        texts = [strategy_to_signature(s) for s in strategies]
        vectors = np.array(self.embedder.embed_documents(texts), dtype="float32")
        self._faiss_dim = vectors.shape[1]
        self._faiss_index = faiss.IndexFlatL2(self._faiss_dim)
        self._faiss_index.add(vectors)
        self._faiss_ids = [s.strategy_id for s in strategies]
        self._save_faiss()
        return len(strategies)

    def count(self) -> int:
        return len(self.list_all())

    # --- FAISS internals ---

    def _upsert_vector(self, strategy: Strategy) -> None:
        """Insert or replace a single strategy's vector."""
        import faiss

        vec = np.array(
            [self.embedder.embed_query(strategy_to_signature(strategy))],
            dtype="float32",
        )

        if self._faiss_index is None:
            self._faiss_dim = vec.shape[1]
            self._faiss_index = faiss.IndexFlatL2(self._faiss_dim)
            self._faiss_ids = []

        if strategy.strategy_id in self._faiss_ids:
            # In-place update is not supported by IndexFlatL2; rebuild is the
            # clean path when a strategy is updated.
            self.rebuild_vector_index()
            return

        self._faiss_index.add(vec)
        self._faiss_ids.append(strategy.strategy_id)
        self._save_faiss()

    def _save_faiss(self) -> None:
        import faiss

        if self._faiss_index is None:
            return
        faiss.write_index(self._faiss_index, str(self._faiss_path))
        self._vec_meta_path.write_text(
            json.dumps(
                {"dim": self._faiss_dim, "ids": self._faiss_ids},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _load_faiss(self) -> bool:
        if not (self._faiss_path.exists() and self._vec_meta_path.exists()):
            return False
        import faiss

        self._faiss_index = faiss.read_index(str(self._faiss_path))
        meta = json.loads(self._vec_meta_path.read_text(encoding="utf-8"))
        self._faiss_dim = meta["dim"]
        self._faiss_ids = meta["ids"]
        return True

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
            # JSON block so parser can round-trip structured metrics losslessly.
            perf_json = json.dumps(s.historical_performance, ensure_ascii=False)
            perf = f"\n## Historical Performance\n```json\n{perf_json}\n```\n"
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

        # Historical performance is stored as a JSON code block.
        history: list[dict] = []
        perf_raw = get_section("Historical Performance")
        if perf_raw:
            perf_clean = perf_raw.replace("```json", "").replace("```", "").strip()
            if perf_clean:
                try:
                    parsed = json.loads(perf_clean)
                    if isinstance(parsed, list):
                        history = parsed
                except json.JSONDecodeError:
                    history = []

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
            historical_performance=history,
            version=int(meta.get("version", 1)),
            created_at=meta.get("created_at", ""),
            parent_id=meta.get("parent_id") if meta.get("parent_id") != "none" else None,
        )
