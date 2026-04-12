"""Ad Simulation Environment — combines user pool, content library, and simulator.

The Agent interacts with this environment to test recommendation strategies.
Zero-cost learning through simulated user feedback.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI

from src.models import (
    ActionType,
    CampaignResult,
    ContentItem,
    RecommendAction,
    UserFeedback,
    UserProfile,
)
from src.simulation.user_simulator import UserSimulator


@dataclass
class AdEnvironment:
    """Simulated ad/recommendation environment."""

    users: list[UserProfile]
    contents: list[ContentItem]
    simulator: UserSimulator
    round_counter: int = 0

    @classmethod
    def create(
        cls,
        users: list[UserProfile],
        contents: list[ContentItem],
        llm: ChatOpenAI,
    ) -> "AdEnvironment":
        return cls(
            users=users,
            contents=contents,
            simulator=UserSimulator(llm),
        )

    def step(self, actions: list[RecommendAction], strategy_name: str | None = None) -> CampaignResult:
        """Execute one round: push recommendations, collect simulated feedback."""
        self.round_counter += 1
        feedbacks = self.simulator.simulate_batch(actions)
        return self._aggregate(feedbacks, actions, strategy_name)

    def get_users(self, n: int | None = None) -> list[UserProfile]:
        """Get a sample of users for this round."""
        if n is None or n >= len(self.users):
            return self.users
        import random
        return random.sample(self.users, n)

    def get_contents(self, category: str | None = None) -> list[ContentItem]:
        """Get contents, optionally filtered by category."""
        if category is None:
            return self.contents
        return [c for c in self.contents if category in (c.category_l1, c.category_l2, c.category_l3)]

    def _aggregate(
        self,
        feedbacks: list[UserFeedback],
        actions: list[RecommendAction],
        strategy_name: str | None,
    ) -> CampaignResult:
        """Aggregate individual feedbacks into campaign-level metrics."""
        total = len(feedbacks)
        clicks = sum(1 for f in feedbacks if f.action != ActionType.SKIP)
        # Complete = watched 60%+ (realistic for short video platforms)
        completes = sum(1 for f in feedbacks if f.watch_ratio >= 0.6)
        likes = sum(1 for f in feedbacks if f.action == ActionType.LIKE)
        shares = sum(1 for f in feedbacks if f.action == ActionType.SHARE)

        trajectory = [
            {
                "user_id": a.user.user_id,
                "item_id": a.content.item_id,
                "targeting_reason": a.targeting_reason,
                "feedback_action": f.action.value,
                "watch_ratio": f.watch_ratio,
                "feedback_reason": f.reason,
            }
            for a, f in zip(actions, feedbacks)
        ]

        return CampaignResult(
            round_id=self.round_counter,
            total_impressions=total,
            clicks=clicks,
            completes=completes,
            likes=likes,
            shares=shares,
            strategy_used=strategy_name,
            trajectory=trajectory,
        )
