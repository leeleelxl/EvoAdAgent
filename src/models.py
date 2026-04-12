"""Core data models for EvoAdAgent."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    UNKNOWN = "unknown"


class ActionType(str, Enum):
    CLICK = "click"
    SKIP = "skip"
    LIKE = "like"
    SHARE = "share"
    COMMENT = "comment"
    FOLLOW = "follow"
    COMPLETE_PLAY = "complete_play"


@dataclass
class UserProfile:
    """Simulated user profile, based on KuaiRec user features."""

    user_id: str
    gender: Gender
    age_range: str  # e.g. "18-24", "25-30"
    province: str
    city: str
    city_level: str  # e.g. "一线", "二线", "三线"
    interests: list[str] = field(default_factory=list)  # e.g. ["宠物", "美食"]
    device_price: str = "mid"  # low | mid | high
    active_degree: str = "medium"  # low | medium | high


@dataclass
class ContentItem:
    """A piece of content (video/ad) to recommend, based on KuaiRec items."""

    item_id: str
    caption: str  # video title + description
    topic_tags: list[str]  # hashtags
    category_l1: str  # e.g. "宠物"
    category_l2: str  # e.g. "宠物日常记录"
    category_l3: str  # e.g. "宠物狗"
    duration_seconds: float = 30.0


@dataclass
class RecommendAction:
    """Agent's recommendation decision for a user-content pair."""

    user: UserProfile
    content: ContentItem
    targeting_reason: str  # why this content for this user
    bid_score: float = 1.0  # priority score


@dataclass
class UserFeedback:
    """Simulated user response to a recommendation."""

    user_id: str
    item_id: str
    action: ActionType
    watch_ratio: float = 0.0  # 0.0 - 1.0, how much of the video was watched
    reason: str = ""  # LLM-generated reason for the action


@dataclass
class CampaignResult:
    """Results from one round of recommendation campaign."""

    round_id: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    total_impressions: int = 0
    clicks: int = 0
    completes: int = 0
    likes: int = 0
    shares: int = 0
    strategy_used: str | None = None
    trajectory: list[dict] = field(default_factory=list)  # full decision trace

    @property
    def ctr(self) -> float:
        return self.clicks / max(self.total_impressions, 1)

    @property
    def completion_rate(self) -> float:
        return self.completes / max(self.total_impressions, 1)

    @property
    def engagement_rate(self) -> float:
        return (self.likes + self.shares) / max(self.total_impressions, 1)


@dataclass
class AdReflection:
    """Structured reflection after a campaign round."""

    round_id: int
    campaign_result: CampaignResult
    what_worked: list[str]
    what_failed: list[str]
    root_causes: list[str]
    improvement_suggestions: list[str]
    key_insight: str


class StrategyType(str, Enum):
    NEW = "new"
    REFINE = "refine"
    MERGE = "merge"


@dataclass
class Strategy:
    """A distilled, reusable recommendation strategy."""

    strategy_id: str
    name: str
    strategy_type: StrategyType
    applicable_scenario: str  # when to use this strategy
    target_audience: str  # who this strategy targets
    content_direction: str  # what kind of content to recommend
    execution_steps: list[str]  # step-by-step how to execute
    expected_effect: str  # expected improvement
    historical_performance: list[dict] = field(default_factory=list)  # past results
    version: int = 1
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    parent_id: str | None = None  # for REFINE/MERGE, link to parent strategy
