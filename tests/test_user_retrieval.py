"""Tests for the find_similar_users tool — confirms L2 FAISS is actually
reachable from the ReAct tool layer."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from src.memory.user_profile import UserProfileStore
from src.models import Gender, UserProfile
from src.tools.user_retrieval import build_find_similar_users_tool


class FakeEmbedder:
    def __init__(self, dim: int = 32):
        self.dim = dim

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t).tolist() for t in texts]

    def _vec(self, text: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self.dim).astype("float32")
        v /= np.linalg.norm(v) + 1e-9
        return v


def _make_user(uid: str, age: str = "25-30", interests=None) -> UserProfile:
    return UserProfile(
        user_id=uid,
        gender=Gender.FEMALE,
        age_range=age,
        province="广东",
        city="广州",
        city_level="一线",
        interests=interests or ["宠物"],
        device_price="mid",
        active_degree="high",
    )


@pytest.fixture()
def populated_store():
    store = UserProfileStore()
    store._embedder = FakeEmbedder(dim=32)
    store.build([_make_user(f"u{i}") for i in range(5)])
    return store


class TestToolSchema:
    def test_tool_has_correct_name(self, populated_store):
        tool = build_find_similar_users_tool(populated_store)
        assert tool.name == "find_similar_users"

    def test_tool_has_docstring(self, populated_store):
        tool = build_find_similar_users_tool(populated_store)
        assert tool.description
        assert "相似" in tool.description


class TestInvocation:
    def _invoke(self, tool, **kwargs) -> str:
        return tool.invoke(kwargs)

    def test_returns_similar_users(self, populated_store):
        tool = build_find_similar_users_tool(populated_store)
        out = self._invoke(tool, user_id="u0", k=3)
        # Should include some but not u0 itself
        assert "u0" not in out.split("\n\n以上")[0] or "相似" in out
        # At least some other users should appear
        other_ids = [f"u{i}" for i in range(1, 5)]
        assert any(uid in out for uid in other_ids)

    def test_excludes_target_user(self, populated_store):
        tool = build_find_similar_users_tool(populated_store)
        out = self._invoke(tool, user_id="u2", k=2)
        # u2 should not appear in the similar-user list section
        similar_section = out.split("以上")[0]
        # u2 may appear in the header line but not in the "- **u2**" entry form
        assert "- **u2**" not in similar_section

    def test_k_respected(self, populated_store):
        tool = build_find_similar_users_tool(populated_store)
        out = self._invoke(tool, user_id="u0", k=2)
        # Count bullet lines
        bullets = [line for line in out.split("\n") if line.strip().startswith("- **")]
        assert len(bullets) == 2

    def test_unknown_user_handled_gracefully(self, populated_store):
        tool = build_find_similar_users_tool(populated_store)
        out = self._invoke(tool, user_id="nonexistent", k=3)
        assert "不在" in out or "nonexistent" in out

    def test_empty_store_returns_informative_message(self):
        empty_store = UserProfileStore()
        tool = build_find_similar_users_tool(empty_store)
        out = tool.invoke({"user_id": "anyone", "k": 3})
        assert "空" in out or "为空" in out

    def test_default_k_is_3(self, populated_store):
        tool = build_find_similar_users_tool(populated_store)
        out = tool.invoke({"user_id": "u0"})  # k omitted
        bullets = [line for line in out.split("\n") if line.strip().startswith("- **")]
        assert len(bullets) == 3


class TestIntegrationWithGetAllTools:
    def test_find_similar_users_registered_when_store_provided(self):
        from src.tools import get_all_tools

        store = UserProfileStore()
        store._embedder = FakeEmbedder(dim=32)
        store.build([_make_user(f"u{i}") for i in range(3)])

        tools = get_all_tools(user_profile_store=store)
        tool_names = {t.name for t in tools}
        assert "find_similar_users" in tool_names

    def test_find_similar_users_absent_when_no_store(self):
        from src.tools import get_all_tools

        tools = get_all_tools()
        tool_names = {t.name for t in tools}
        assert "find_similar_users" not in tool_names
