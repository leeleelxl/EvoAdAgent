"""Tests for the probabilistic UserSimulator sampling logic.

Behavior under test: given LLM-estimated interaction probabilities, the
Python side should produce Bernoulli-sampled outcomes whose aggregate matches
the probabilities (no 100%-click saturation).
"""

from __future__ import annotations

import json

from src.models import ActionType
from src.simulation.user_simulator import UserSimulator


def _llm_response(click_prob, complete_prob=0.3, like_prob=0.1, share_prob=0.02):
    return json.dumps(
        {
            "click_prob": click_prob,
            "complete_prob": complete_prob,
            "like_prob": like_prob,
            "share_prob": share_prob,
            "match_level": "medium",
            "reason": "test",
        }
    )


class TestDeterminism:
    def test_same_user_item_always_same_outcome(self):
        """Deterministic RNG → reproducible outcomes for fair A/B comparison."""
        resp = _llm_response(0.5)
        a = UserSimulator._parse_and_sample(resp, "u1", "c1")
        b = UserSimulator._parse_and_sample(resp, "u1", "c1")
        assert a.action == b.action
        assert a.watch_ratio == b.watch_ratio

    def test_different_pairs_get_different_seeds(self):
        resp = _llm_response(0.5)
        a = UserSimulator._seed_rng("u1", "c1").random()
        b = UserSimulator._seed_rng("u1", "c2").random()
        c = UserSimulator._seed_rng("u2", "c1").random()
        assert len({a, b, c}) == 3


class TestProbabilityCalibration:
    def test_high_click_prob_mostly_clicks(self):
        """With click_prob=0.9, aggregate click rate should be close to 0.9."""
        resp = _llm_response(0.9)
        clicks = sum(
            UserSimulator._parse_and_sample(resp, f"u{i}", f"c{i}").action != ActionType.SKIP
            for i in range(200)
        )
        assert 0.75 < clicks / 200 < 1.0

    def test_low_click_prob_mostly_skips(self):
        resp = _llm_response(0.05)
        clicks = sum(
            UserSimulator._parse_and_sample(resp, f"u{i}", f"c{i}").action != ActionType.SKIP
            for i in range(200)
        )
        assert 0.0 <= clicks / 200 < 0.2

    def test_zero_click_prob_never_clicks(self):
        resp = _llm_response(0.0)
        for i in range(50):
            fb = UserSimulator._parse_and_sample(resp, f"u{i}", f"c{i}")
            assert fb.action == ActionType.SKIP
            assert fb.watch_ratio == 0.0

    def test_click_prob_1_saturates(self):
        """Even at click_prob=1.0 we expect 100% click — but not all complete."""
        resp = _llm_response(1.0, complete_prob=0.5)
        fbs = [
            UserSimulator._parse_and_sample(resp, f"u{i}", f"c{i}") for i in range(100)
        ]
        skips = sum(1 for f in fbs if f.action == ActionType.SKIP)
        completes = sum(1 for f in fbs if f.watch_ratio >= 0.9)
        assert skips == 0
        # At complete_prob=0.5 with noise, not all completes
        assert completes < 100


class TestWatchRatioDistribution:
    def test_high_complete_prob_pushes_watch_ratio_up(self):
        high = _llm_response(click_prob=1.0, complete_prob=0.9)
        low = _llm_response(click_prob=1.0, complete_prob=0.1)
        high_ratios = [
            UserSimulator._parse_and_sample(high, f"u{i}", f"c{i}").watch_ratio
            for i in range(100)
        ]
        low_ratios = [
            UserSimulator._parse_and_sample(low, f"u{i}", f"c{i}").watch_ratio
            for i in range(100)
        ]
        assert sum(high_ratios) / 100 > sum(low_ratios) / 100

    def test_watch_ratio_in_range(self):
        resp = _llm_response(1.0, complete_prob=0.5)
        for i in range(50):
            fb = UserSimulator._parse_and_sample(resp, f"u{i}", f"c{i}")
            assert 0.0 <= fb.watch_ratio <= 1.0


class TestEngagementGating:
    def test_like_requires_meaningful_watch(self):
        """User who barely watched shouldn't like — gated by watch_ratio >= 0.3."""
        resp = _llm_response(click_prob=1.0, complete_prob=0.0, like_prob=1.0)
        # complete_prob=0 → most watch_ratio < 0.3 → most likes suppressed
        likes = 0
        total = 100
        for i in range(total):
            fb = UserSimulator._parse_and_sample(resp, f"u{i}", f"c{i}")
            if fb.action == ActionType.LIKE:
                likes += 1
        # Gate should prevent saturation at 100%
        assert likes / total < 0.95


class TestParseErrors:
    def test_invalid_json_returns_skip(self):
        fb = UserSimulator._parse_and_sample("not json", "u1", "c1")
        assert fb.action == ActionType.SKIP
        assert "parse error" in fb.reason

    def test_code_fence_is_stripped(self):
        resp = f"```json\n{_llm_response(0.5)}\n```"
        fb = UserSimulator._parse_and_sample(resp, "u1", "c1")
        # Either click or skip, but not a parse error fallback
        assert "parse error" not in fb.reason

    def test_extra_fields_ignored(self):
        resp = json.dumps(
            {
                "click_prob": 0.5,
                "complete_prob": 0.3,
                "like_prob": 0.1,
                "share_prob": 0.02,
                "match_level": "medium",
                "reason": "t",
                "ignored_field": "extra",
            }
        )
        fb = UserSimulator._parse_and_sample(resp, "u1", "c1")
        assert "parse error" not in fb.reason

    def test_out_of_range_probs_clipped(self):
        resp = json.dumps(
            {
                "click_prob": 1.5,
                "complete_prob": -0.2,
                "like_prob": 0.1,
                "share_prob": 0.02,
                "reason": "t",
            }
        )
        # Should not raise; 1.5 clipped to 1.0 → always clicks
        fb = UserSimulator._parse_and_sample(resp, "u1", "c1")
        assert fb.action != ActionType.SKIP
