"""Tests for core data models."""

import pytest
from src.models import (
    ActionType,
    AdReflection,
    CampaignResult,
    ContentItem,
    Gender,
    RecommendAction,
    Strategy,
    StrategyType,
    UserFeedback,
    UserProfile,
)


# ---- UserProfile ----


class TestUserProfile:
    def test_create_user_profile_with_required_fields(self):
        user = UserProfile(
            user_id="u001",
            gender=Gender.MALE,
            age_range="18-24",
            province="广东",
            city="广州",
            city_level="一线",
        )
        assert user.user_id == "u001"
        assert user.gender == Gender.MALE
        assert user.age_range == "18-24"
        assert user.province == "广东"
        assert user.city == "广州"
        assert user.city_level == "一线"

    def test_user_profile_default_values(self):
        user = UserProfile(
            user_id="u002",
            gender=Gender.FEMALE,
            age_range="25-30",
            province="北京",
            city="北京",
            city_level="一线",
        )
        assert user.interests == []
        assert user.device_price == "mid"
        assert user.active_degree == "medium"

    def test_user_profile_with_custom_optional_fields(self):
        user = UserProfile(
            user_id="u003",
            gender=Gender.UNKNOWN,
            age_range="31-40",
            province="浙江",
            city="杭州",
            city_level="二线",
            interests=["美食", "旅行"],
            device_price="high",
            active_degree="high",
        )
        assert user.interests == ["美食", "旅行"]
        assert user.device_price == "high"
        assert user.active_degree == "high"


# ---- ContentItem ----


class TestContentItem:
    def test_create_content_item(self):
        content = ContentItem(
            item_id="v001",
            caption="测试视频 #宠物",
            topic_tags=["宠物", "搞笑"],
            category_l1="宠物",
            category_l2="宠物日常记录",
            category_l3="宠物狗",
        )
        assert content.item_id == "v001"
        assert content.caption == "测试视频 #宠物"
        assert content.topic_tags == ["宠物", "搞笑"]
        assert content.category_l1 == "宠物"
        assert content.category_l2 == "宠物日常记录"
        assert content.category_l3 == "宠物狗"

    def test_content_item_default_duration(self):
        content = ContentItem(
            item_id="v002",
            caption="test",
            topic_tags=[],
            category_l1="科技",
            category_l2="数码测评",
            category_l3="手机",
        )
        assert content.duration_seconds == 30.0

    def test_content_item_custom_duration(self):
        content = ContentItem(
            item_id="v003",
            caption="长视频",
            topic_tags=["教程"],
            category_l1="美食",
            category_l2="美食教程",
            category_l3="家常菜",
            duration_seconds=180.0,
        )
        assert content.duration_seconds == 180.0


# ---- RecommendAction ----


class TestRecommendAction:
    def test_create_recommend_action(self):
        user = UserProfile("u001", Gender.MALE, "18-24", "广东", "广州", "一线")
        content = ContentItem("v001", "caption", ["tag"], "宠物", "l2", "l3")
        action = RecommendAction(
            user=user,
            content=content,
            targeting_reason="用户喜欢宠物",
        )
        assert action.user.user_id == "u001"
        assert action.content.item_id == "v001"
        assert action.targeting_reason == "用户喜欢宠物"
        assert action.bid_score == 1.0

    def test_recommend_action_custom_bid(self):
        user = UserProfile("u001", Gender.MALE, "18-24", "广东", "广州", "一线")
        content = ContentItem("v001", "caption", ["tag"], "宠物", "l2", "l3")
        action = RecommendAction(
            user=user,
            content=content,
            targeting_reason="高优先级",
            bid_score=5.0,
        )
        assert action.bid_score == 5.0


# ---- UserFeedback ----


class TestUserFeedback:
    def test_create_user_feedback(self):
        fb = UserFeedback(
            user_id="u001",
            item_id="v001",
            action=ActionType.CLICK,
        )
        assert fb.user_id == "u001"
        assert fb.item_id == "v001"
        assert fb.action == ActionType.CLICK
        assert fb.watch_ratio == 0.0
        assert fb.reason == ""

    def test_user_feedback_with_watch_ratio(self):
        fb = UserFeedback(
            user_id="u002",
            item_id="v002",
            action=ActionType.COMPLETE_PLAY,
            watch_ratio=1.0,
            reason="内容有趣",
        )
        assert fb.watch_ratio == 1.0
        assert fb.reason == "内容有趣"

    def test_action_type_values(self):
        assert ActionType.CLICK.value == "click"
        assert ActionType.SKIP.value == "skip"
        assert ActionType.LIKE.value == "like"
        assert ActionType.SHARE.value == "share"
        assert ActionType.COMMENT.value == "comment"
        assert ActionType.FOLLOW.value == "follow"
        assert ActionType.COMPLETE_PLAY.value == "complete_play"


# ---- CampaignResult ----


class TestCampaignResult:
    def test_create_campaign_result_defaults(self):
        result = CampaignResult(round_id=1)
        assert result.round_id == 1
        assert result.total_impressions == 0
        assert result.clicks == 0
        assert result.completes == 0
        assert result.likes == 0
        assert result.shares == 0
        assert result.strategy_used is None
        assert result.trajectory == []
        assert result.timestamp  # should be auto-generated

    def test_ctr_calculation(self):
        result = CampaignResult(round_id=1, total_impressions=100, clicks=25)
        assert result.ctr == pytest.approx(0.25)

    def test_ctr_zero_impressions(self):
        result = CampaignResult(round_id=1, total_impressions=0, clicks=0)
        assert result.ctr == 0.0

    def test_completion_rate_calculation(self):
        result = CampaignResult(round_id=1, total_impressions=200, completes=50)
        assert result.completion_rate == pytest.approx(0.25)

    def test_completion_rate_zero_impressions(self):
        result = CampaignResult(round_id=1, total_impressions=0, completes=0)
        assert result.completion_rate == 0.0

    def test_engagement_rate_calculation(self):
        result = CampaignResult(
            round_id=1, total_impressions=100, likes=10, shares=5
        )
        assert result.engagement_rate == pytest.approx(0.15)

    def test_engagement_rate_zero_impressions(self):
        result = CampaignResult(round_id=1, total_impressions=0, likes=0, shares=0)
        assert result.engagement_rate == 0.0

    def test_full_campaign_result(self):
        result = CampaignResult(
            round_id=3,
            total_impressions=500,
            clicks=100,
            completes=80,
            likes=30,
            shares=20,
            strategy_used="兴趣匹配策略",
            trajectory=[{"step": 1, "action": "analyze"}],
        )
        assert result.ctr == pytest.approx(0.2)
        assert result.completion_rate == pytest.approx(0.16)
        assert result.engagement_rate == pytest.approx(0.1)
        assert result.strategy_used == "兴趣匹配策略"
        assert len(result.trajectory) == 1


# ---- Strategy ----


class TestStrategy:
    def test_create_strategy(self):
        s = Strategy(
            strategy_id="s001",
            name="兴趣匹配策略",
            strategy_type=StrategyType.NEW,
            applicable_scenario="宠物内容推荐",
            target_audience="18-30岁宠物爱好者",
            content_direction="宠物日常和护理内容",
            execution_steps=["分析用户兴趣", "匹配宠物内容", "排序推荐"],
            expected_effect="CTR提升10%",
        )
        assert s.strategy_id == "s001"
        assert s.name == "兴趣匹配策略"
        assert s.strategy_type == StrategyType.NEW
        assert len(s.execution_steps) == 3
        assert s.version == 1
        assert s.parent_id is None
        assert s.historical_performance == []
        assert s.created_at  # auto-generated

    def test_strategy_type_values(self):
        assert StrategyType.NEW.value == "new"
        assert StrategyType.REFINE.value == "refine"
        assert StrategyType.MERGE.value == "merge"

    def test_strategy_with_parent(self):
        s = Strategy(
            strategy_id="s002",
            name="优化版兴趣匹配",
            strategy_type=StrategyType.REFINE,
            applicable_scenario="宠物内容推荐",
            target_audience="年轻女性宠物爱好者",
            content_direction="宠物萌宠视频",
            execution_steps=["步骤1", "步骤2"],
            expected_effect="CTR提升15%",
            parent_id="s001",
            version=2,
        )
        assert s.parent_id == "s001"
        assert s.version == 2
        assert s.strategy_type == StrategyType.REFINE


# ---- Gender enum ----


class TestGenderEnum:
    def test_gender_values(self):
        assert Gender.MALE.value == "male"
        assert Gender.FEMALE.value == "female"
        assert Gender.UNKNOWN.value == "unknown"

    def test_gender_is_str(self):
        assert isinstance(Gender.MALE, str)
        assert Gender.MALE == "male"
