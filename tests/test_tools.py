"""Tests for the ad tool modules — focus on bidding and performance
(new functionality), plus sanity on aggregated get_all_tools()."""

from __future__ import annotations

import json

import pytest

from src.tools import (
    analyze_audience,
    analyze_content_pool,
    evaluate_performance,
    generate_recommendation,
    get_all_tools,
    load_strategy,
    match_user_content,
    set_bid_strategy,
)


def _invoke(tool, **kwargs) -> str:
    """LangChain @tool expects .invoke(dict_of_args)."""
    return tool.invoke(kwargs)


class TestToolRegistry:
    def test_get_all_tools_count(self):
        assert len(get_all_tools()) == 7

    def test_all_tools_have_names(self):
        for t in get_all_tools():
            assert t.name

    def test_get_all_tools_unique_names(self):
        names = [t.name for t in get_all_tools()]
        assert len(names) == len(set(names))


class TestBidding:
    def test_balanced_mode_defaults(self):
        out = _invoke(
            set_bid_strategy,
            audience_size=10000,
            competition_level="medium",
        )
        d = json.loads(out)
        assert d["mode"] == "balanced"
        assert d["bid_score"] == pytest.approx(1.0)

    def test_aggressive_high_competition_raises_score(self):
        out = _invoke(
            set_bid_strategy,
            audience_size=10000,
            competition_level="high",
            mode="aggressive",
        )
        d = json.loads(out)
        assert d["bid_score"] > 1.5

    def test_small_audience_bumps_bid(self):
        out = _invoke(
            set_bid_strategy,
            audience_size=500,
            competition_level="medium",
            mode="balanced",
        )
        d = json.loads(out)
        assert d["bid_score"] > 1.0  # size_factor 1.2 multiplier

    def test_large_audience_reduces_bid(self):
        out = _invoke(
            set_bid_strategy,
            audience_size=500_000,
            competition_level="medium",
            mode="balanced",
        )
        d = json.loads(out)
        assert d["bid_score"] < 1.0

    def test_invalid_mode_falls_back_to_balanced(self):
        out = _invoke(
            set_bid_strategy,
            audience_size=10000,
            competition_level="medium",
            mode="not_a_mode",
        )
        d = json.loads(out)
        assert d["mode"] == "balanced"

    def test_rationale_included(self):
        out = _invoke(
            set_bid_strategy,
            audience_size=10000,
            competition_level="low",
        )
        d = json.loads(out)
        assert "rationale" in d and d["rationale"]


class TestPerformance:
    def test_metrics_computed(self):
        payload = json.dumps(
            {
                "total_impressions": 1000,
                "clicks": 100,
                "completes": 400,
                "likes": 50,
                "shares": 30,
            }
        )
        out = _invoke(evaluate_performance, campaign_result_json=payload)
        assert "CTR:" in out and "10.00%" in out
        assert "完播率:" in out and "40.00%" in out
        assert "互动率:" in out and "8.00%" in out

    def test_high_performance_verdict_excellent(self):
        payload = json.dumps(
            {
                "total_impressions": 100,
                "clicks": 50,
                "completes": 60,
                "likes": 30,
                "shares": 20,
            }
        )
        out = _invoke(evaluate_performance, campaign_result_json=payload)
        assert "优秀" in out

    def test_low_performance_verdict_needs_work(self):
        payload = json.dumps(
            {
                "total_impressions": 1000,
                "clicks": 5,
                "completes": 10,
                "likes": 2,
                "shares": 1,
            }
        )
        out = _invoke(evaluate_performance, campaign_result_json=payload)
        assert "待优化" in out

    def test_invalid_json_returns_error(self):
        out = _invoke(evaluate_performance, campaign_result_json="not json")
        assert "解析错误" in out

    def test_zero_impressions_no_division_error(self):
        payload = json.dumps({"total_impressions": 0, "clicks": 0})
        out = _invoke(evaluate_performance, campaign_result_json=payload)
        # Should not crash; CTR should be 0
        assert "0.00%" in out


class TestAudienceAnalyzer:
    def test_analyze_audience_basic(self):
        users = json.dumps(
            [
                {"gender": "female", "age_range": "25-30", "city_level": "一线",
                 "interests": ["宠物", "美食"]},
                {"gender": "male", "age_range": "18-24", "city_level": "二线",
                 "interests": ["游戏"]},
            ]
        )
        out = _invoke(analyze_audience, user_profiles_json=users)
        assert "共 2 人" in out
        assert "宠物" in out

    def test_analyze_audience_empty(self):
        out = _invoke(analyze_audience, user_profiles_json="[]")
        assert "空" in out

    def test_analyze_content_pool_basic(self):
        contents = json.dumps(
            [
                {"item_id": "c1", "category_l1": "宠物", "topic_tags": ["猫", "日常"]},
                {"item_id": "c2", "category_l1": "美食", "topic_tags": ["早餐"]},
            ]
        )
        out = _invoke(analyze_content_pool, contents_json=contents)
        assert "共 2 条" in out
        assert "宠物" in out


class TestTargeting:
    def test_match_user_content_scores_interest_overlap(self):
        user = json.dumps({"user_id": "u1", "interests": ["宠物"]})
        contents = json.dumps(
            [
                {"item_id": "c1", "caption": "猫咪日常", "category_l1": "宠物",
                 "topic_tags": ["猫", "宠物"]},
                {"item_id": "c2", "caption": "游戏攻略", "category_l1": "游戏",
                 "topic_tags": ["游戏"]},
            ]
        )
        out = _invoke(match_user_content, user_json=user, contents_json=contents)
        # c1 should appear ranked higher than c2
        c1_pos = out.find("c1")
        c2_pos = out.find("c2")
        assert c1_pos >= 0 and (c2_pos < 0 or c1_pos < c2_pos)

    def test_generate_recommendation_returns_json(self):
        out = _invoke(
            generate_recommendation,
            user_id="u1",
            item_id="c1",
            targeting_reason="兴趣匹配",
        )
        d = json.loads(out)
        assert d["user_id"] == "u1"
        assert d["item_id"] == "c1"
        assert d["targeting_reason"] == "兴趣匹配"


class TestCreativeGenerator:
    def test_load_strategy_placeholder(self):
        out = _invoke(load_strategy, strategy_info="宠物精准定向")
        assert "宠物精准定向" in out
