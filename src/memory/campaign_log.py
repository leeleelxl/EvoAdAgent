"""L1: Campaign Log — stores execution trajectories and results in SQLite."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from src.models import AdReflection, CampaignResult


class CampaignLog:
    """Persistent storage for campaign execution history."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS campaigns (
                    round_id INTEGER PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    total_impressions INTEGER,
                    clicks INTEGER,
                    completes INTEGER,
                    likes INTEGER,
                    shares INTEGER,
                    ctr REAL,
                    completion_rate REAL,
                    engagement_rate REAL,
                    strategy_used TEXT,
                    trajectory TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reflections (
                    round_id INTEGER PRIMARY KEY,
                    what_worked TEXT,
                    what_failed TEXT,
                    root_causes TEXT,
                    improvement_suggestions TEXT,
                    key_insight TEXT,
                    FOREIGN KEY (round_id) REFERENCES campaigns(round_id)
                )
            """)

    def _conn(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.db_path)

    def save_campaign(self, result: CampaignResult):
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO campaigns
                   (round_id, timestamp, total_impressions, clicks, completes,
                    likes, shares, ctr, completion_rate, engagement_rate,
                    strategy_used, trajectory)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    result.round_id,
                    result.timestamp,
                    result.total_impressions,
                    result.clicks,
                    result.completes,
                    result.likes,
                    result.shares,
                    result.ctr,
                    result.completion_rate,
                    result.engagement_rate,
                    result.strategy_used,
                    json.dumps(result.trajectory, ensure_ascii=False),
                ),
            )

    def save_reflection(self, reflection: AdReflection):
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO reflections
                   (round_id, what_worked, what_failed, root_causes,
                    improvement_suggestions, key_insight)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    reflection.round_id,
                    json.dumps(reflection.what_worked, ensure_ascii=False),
                    json.dumps(reflection.what_failed, ensure_ascii=False),
                    json.dumps(reflection.root_causes, ensure_ascii=False),
                    json.dumps(reflection.improvement_suggestions, ensure_ascii=False),
                    reflection.key_insight,
                ),
            )

    def get_recent_campaigns(self, n: int = 10) -> list[dict]:
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM campaigns ORDER BY round_id DESC LIMIT ?", (n,)
            ).fetchall()
            return [dict(row) for row in rows]

    def get_recent_reflections(self, n: int = 5) -> list[dict]:
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM reflections ORDER BY round_id DESC LIMIT ?", (n,)
            ).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                for key in ("what_worked", "what_failed", "root_causes", "improvement_suggestions"):
                    if d[key]:
                        d[key] = json.loads(d[key])
                results.append(d)
            return results

    def get_evolution_curve(self) -> list[dict]:
        """Get CTR/engagement progression across all rounds."""
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT round_id, ctr, completion_rate, engagement_rate, strategy_used "
                "FROM campaigns ORDER BY round_id"
            ).fetchall()
            return [dict(row) for row in rows]
