"""Tests for L2 user profile FAISS store.

Uses a fake deterministic embedder to avoid real API calls — keeps unit tests
offline and cheap. Real-API smoke is covered by examples/demo_user_profile.py.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from src.memory.user_profile import UserProfileStore, user_to_persona_text
from src.models import Gender, UserProfile


class FakeEmbedder:
    """Deterministic text→vector mapper. Same text always maps to the same vector.

    Similar strings share overlapping hash seeds so L2 distance is meaningful.
    """

    def __init__(self, dim: int = 32):
        self.dim = dim

    def embed_query(self, text: str) -> list[float]:
        return self._hash_to_vec(text).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_to_vec(t).tolist() for t in texts]

    def _hash_to_vec(self, text: str) -> np.ndarray:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(h[:4], "big")
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self.dim).astype("float32")
        v /= np.linalg.norm(v) + 1e-9
        return v


def make_user(
    uid: str,
    gender: Gender = Gender.FEMALE,
    age: str = "25-30",
    city: str = "广州",
    level: str = "一线",
    interests: list[str] | None = None,
) -> UserProfile:
    if interests is None:
        interests = ["短视频"]
    return UserProfile(
        user_id=uid,
        gender=gender,
        age_range=age,
        province="广东",
        city=city,
        city_level=level,
        interests=interests,
        device_price="mid",
        active_degree="high",
    )


@pytest.fixture()
def store():
    s = UserProfileStore()
    s._embedder = FakeEmbedder(dim=32)
    return s


class TestPersonaText:
    def test_basic_persona(self):
        user = make_user("u1", Gender.FEMALE, "25-30", "广州", "一线", ["宠物", "美食"])
        text = user_to_persona_text(user)
        assert "女性" in text
        assert "25-30" in text
        assert "广州" in text
        assert "一线" in text
        assert "宠物" in text and "美食" in text

    def test_male_persona(self):
        user = make_user("u1", gender=Gender.MALE)
        assert "男性" in user_to_persona_text(user)

    def test_empty_interests_defaults_to_generic(self):
        user = make_user("u1", interests=[])
        assert "综合" in user_to_persona_text(user)


class TestBuildAndSearch:
    def test_build_empty_raises(self, store):
        with pytest.raises(ValueError):
            store.build([])

    def test_build_indexes_all_users(self, store):
        users = [make_user(f"u{i}") for i in range(5)]
        store.build(users)
        assert store.count() == 5

    def test_search_returns_k_results(self, store):
        users = [make_user(f"u{i}") for i in range(10)]
        store.build(users)

        results = store.search_similar(users[0], k=3)
        assert len(results) == 3
        for user, dist in results:
            assert isinstance(user, UserProfile)
            assert dist >= 0.0

    def test_search_k_capped_by_index_size(self, store):
        users = [make_user(f"u{i}") for i in range(3)]
        store.build(users)

        results = store.search_similar(users[0], k=100)
        assert len(results) == 3

    def test_search_by_text_query(self, store):
        users = [make_user(f"u{i}") for i in range(5)]
        store.build(users)

        results = store.search_by_text("年轻女性用户", k=2)
        assert len(results) == 2

    def test_search_on_empty_index_returns_empty(self, store):
        results = store.search_by_text("任意查询")
        assert results == []

    def test_identical_user_is_nearest_to_itself(self, store):
        users = [
            make_user("u1", Gender.FEMALE, "25-30", "广州", "一线", ["宠物"]),
            make_user("u2", Gender.MALE, "31-40", "乌鲁木齐", "四线", ["游戏"]),
            make_user("u3", Gender.FEMALE, "18-24", "北京", "一线", ["美食"]),
        ]
        store.build(users)

        results = store.search_similar(users[0], k=1)
        assert results[0][0].user_id == "u1"
        assert results[0][1] == pytest.approx(0.0, abs=1e-4)


class TestIncrementalAdd:
    def test_add_to_empty_store(self, store):
        store.add(make_user("u1"))
        assert store.count() == 1

    def test_add_is_idempotent(self, store):
        u = make_user("u1")
        store.add(u)
        store.add(u)
        assert store.count() == 1

    def test_add_after_build(self, store):
        store.build([make_user(f"u{i}") for i in range(3)])
        store.add(make_user("u_new"))
        assert store.count() == 4
        assert store.get("u_new") is not None


class TestGetAndCount:
    def test_get_returns_stored_user(self, store):
        u = make_user("u1", interests=["宠物", "美食"])
        store.build([u])
        loaded = store.get("u1")
        assert loaded is not None
        assert loaded.interests == ["宠物", "美食"]

    def test_get_missing_returns_none(self, store):
        assert store.get("nope") is None

    def test_count_before_build(self, store):
        assert store.count() == 0


class TestPersistence:
    def test_save_and_load_roundtrip(self, tmp_path, store):
        users = [make_user(f"u{i}") for i in range(5)]
        store.build(users)
        store.save(tmp_path)

        reopened = UserProfileStore(persist_dir=tmp_path)
        reopened._embedder = FakeEmbedder(dim=32)
        assert reopened.load() is True
        assert reopened.count() == 5
        assert reopened.get("u0") is not None
        assert reopened.get("u0").user_id == "u0"

    def test_load_missing_dir_returns_false(self, tmp_path):
        s = UserProfileStore(persist_dir=tmp_path / "nonexistent")
        assert s.load() is False

    def test_save_without_build_raises(self, tmp_path, store):
        with pytest.raises(RuntimeError):
            store.save(tmp_path)

    def test_search_still_works_after_reload(self, tmp_path, store):
        users = [make_user(f"u{i}") for i in range(5)]
        store.build(users)
        store.save(tmp_path)

        reopened = UserProfileStore(persist_dir=tmp_path)
        reopened._embedder = FakeEmbedder(dim=32)
        reopened.load()

        results = reopened.search_similar(users[0], k=1)
        assert len(results) == 1
        assert results[0][0].user_id == "u0"
