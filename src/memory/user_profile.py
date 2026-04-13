"""L2: User Profile vector store — FAISS-based semantic search over user personas.

Converts structured UserProfile objects to Chinese persona text, embeds them
via an OpenAI-compatible embedding API, and indexes them in FAISS for similar
user retrieval. Used by Ad Executor for cold-start targeting reasoning.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

# Must be set before importing faiss: on macOS, faiss and torch both link
# libomp, and importing both without this causes the process to abort.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss  # noqa: E402
import numpy as np  # noqa: E402

from src.config import LLMConfig
from src.llm_factory import create_embeddings
from src.models import Gender, UserProfile


def user_to_persona_text(user: UserProfile) -> str:
    """Convert a UserProfile to a Chinese natural-language persona string.

    Example output:
        "25-30岁女性用户，来自三线城市广州，使用中档手机，活跃度高，兴趣：短视频、游戏"
    """
    gender_cn = {Gender.MALE: "男性", Gender.FEMALE: "女性", Gender.UNKNOWN: "未知性别"}[user.gender]
    device_cn = {"low": "低端手机", "mid": "中档手机", "high": "高端手机"}.get(
        user.device_price, "中档手机"
    )
    active_cn = {"low": "活跃度低", "medium": "活跃度中等", "high": "活跃度高"}.get(
        user.active_degree, "活跃度中等"
    )
    interests = "、".join(user.interests) if user.interests else "综合"

    return (
        f"{user.age_range}岁{gender_cn}用户，来自{user.city_level}城市{user.city}，"
        f"使用{device_cn}，{active_cn}，兴趣：{interests}"
    )


class UserProfileStore:
    """FAISS-backed vector store for UserProfile semantic retrieval."""

    def __init__(
        self,
        emb_config: LLMConfig | None = None,
        persist_dir: Path | str | None = None,
    ):
        self.emb_config = emb_config or LLMConfig(
            provider="qwen", model="text-embedding-v2"
        )
        self._embedder = None
        self._index: faiss.Index | None = None
        self._dim: int | None = None
        # Ordered list: FAISS row i ↔ user_ids[i]
        self._user_ids: list[str] = []
        self._users: dict[str, UserProfile] = {}

        self.persist_dir = Path(persist_dir) if persist_dir else None
        if self.persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = create_embeddings(self.emb_config)
        return self._embedder

    def build(self, users: list[UserProfile], batch_size: int = 25) -> None:
        # Qwen's text-embedding-v2 caps at 25 per request; OpenAI tolerates more.
        """Build the FAISS index from scratch over the given users."""
        if not users:
            raise ValueError("Cannot build index from empty user list")

        texts = [user_to_persona_text(u) for u in users]
        vectors = self._embed_batch(texts, batch_size)
        self._dim = vectors.shape[1]
        self._index = faiss.IndexFlatL2(self._dim)
        self._index.add(vectors)

        self._user_ids = [u.user_id for u in users]
        self._users = {u.user_id: u for u in users}

    def add(self, user: UserProfile) -> None:
        """Add a single user to the existing index."""
        if user.user_id in self._users:
            return  # idempotent
        vec = np.array([self._embed_one(user_to_persona_text(user))], dtype="float32")
        if self._index is None:
            self._dim = vec.shape[1]
            self._index = faiss.IndexFlatL2(self._dim)
        self._index.add(vec)
        self._user_ids.append(user.user_id)
        self._users[user.user_id] = user

    def search_similar(
        self, query_user: UserProfile, k: int = 5
    ) -> list[tuple[UserProfile, float]]:
        """Find k users most similar to the query user (by persona embedding)."""
        return self.search_by_text(user_to_persona_text(query_user), k)

    def search_by_text(self, text: str, k: int = 5) -> list[tuple[UserProfile, float]]:
        """Natural-language query, e.g. '25-35岁女性喜欢美食'."""
        if self._index is None or self._index.ntotal == 0:
            return []
        vec = np.array([self._embed_one(text)], dtype="float32")
        k = min(k, self._index.ntotal)
        distances, indices = self._index.search(vec, k)
        out: list[tuple[UserProfile, float]] = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self._user_ids):
                continue
            uid = self._user_ids[idx]
            out.append((self._users[uid], float(dist)))
        return out

    def get(self, user_id: str) -> UserProfile | None:
        return self._users.get(user_id)

    def count(self) -> int:
        return self._index.ntotal if self._index else 0

    def save(self, path: Path | str | None = None) -> Path:
        """Persist FAISS index + user metadata to disk."""
        target = Path(path) if path else self.persist_dir
        if target is None:
            raise ValueError("No persist path given and no persist_dir configured")
        target.mkdir(parents=True, exist_ok=True)

        if self._index is None:
            raise RuntimeError("Cannot save — index not built yet")

        faiss.write_index(self._index, str(target / "users.faiss"))
        meta = {
            "dim": self._dim,
            "user_ids": self._user_ids,
            "users": {uid: _user_to_dict(u) for uid, u in self._users.items()},
        }
        (target / "users_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return target

    def load(self, path: Path | str | None = None) -> bool:
        """Load FAISS index + metadata from disk. Returns True if loaded."""
        target = Path(path) if path else self.persist_dir
        if target is None:
            return False

        index_path = target / "users.faiss"
        meta_path = target / "users_meta.json"
        if not index_path.exists() or not meta_path.exists():
            return False

        self._index = faiss.read_index(str(index_path))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self._dim = meta["dim"]
        self._user_ids = meta["user_ids"]
        self._users = {uid: _user_from_dict(d) for uid, d in meta["users"].items()}
        return True

    # --- Internals ---

    def _embed_batch(self, texts: list[str], batch_size: int) -> np.ndarray:
        """Embed texts in batches (most providers cap at 25 inputs per request)."""
        vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            vectors.extend(self.embedder.embed_documents(chunk))
        return np.array(vectors, dtype="float32")

    def _embed_one(self, text: str) -> list[float]:
        return self.embedder.embed_query(text)


def _user_to_dict(u: UserProfile) -> dict:
    d = asdict(u)
    d["gender"] = u.gender.value
    return d


def _user_from_dict(d: dict) -> UserProfile:
    return UserProfile(
        user_id=d["user_id"],
        gender=Gender(d["gender"]),
        age_range=d["age_range"],
        province=d["province"],
        city=d["city"],
        city_level=d["city_level"],
        interests=list(d.get("interests", [])),
        device_price=d.get("device_price", "mid"),
        active_degree=d.get("active_degree", "medium"),
    )
