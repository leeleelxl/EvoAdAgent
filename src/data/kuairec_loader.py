"""KuaiRec dataset loader — converts real Kuaishou data into EvoAdAgent models.

Loads user profiles and video content from KuaiRec CSV files and converts
them into UserProfile and ContentItem objects for the simulation environment.

Data source: https://kuairec.com/ (Kuaishou recommendation dataset)
"""

from __future__ import annotations

import ast
import json
import random
from pathlib import Path

import pandas as pd

from src.models import ContentItem, Gender, UserProfile


class KuaiRecLoader:
    """Loads and converts KuaiRec dataset into EvoAdAgent models."""

    def __init__(self, data_dir: Path | str):
        self.data_dir = Path(data_dir)
        self._users_df: pd.DataFrame | None = None
        self._content_df: pd.DataFrame | None = None

    def load_users(self, n: int | None = None, seed: int = 42) -> list[UserProfile]:
        """Load user profiles from user_features_raw.csv.

        Args:
            n: Number of users to load. None = all.
            seed: Random seed for sampling.
        """
        df = self._load_users_df()
        if n and n < len(df):
            df = df.sample(n=n, random_state=seed)

        users = []
        for _, row in df.iterrows():
            user = self._row_to_user(row)
            if user:
                users.append(user)
        return users

    def load_contents(self, n: int | None = None, category: str | None = None) -> list[ContentItem]:
        """Load video content from kuairec_caption_category.csv.

        Args:
            n: Number of contents to load. None = all.
            category: Filter by first-level category name (e.g. "美食").
        """
        df = self._load_content_df()
        if category:
            df = df[df["first_level_category_name"] == category]
        if n and n < len(df):
            df = df.sample(n=n, random_state=42)

        contents = []
        for _, row in df.iterrows():
            item = self._row_to_content(row)
            if item:
                contents.append(item)
        return contents

    def get_categories(self) -> list[str]:
        """Get all unique first-level category names."""
        df = self._load_content_df()
        return sorted(df["first_level_category_name"].dropna().unique().tolist())

    def get_stats(self) -> dict:
        """Get dataset statistics."""
        users_df = self._load_users_df()
        content_df = self._load_content_df()
        return {
            "total_users": len(users_df),
            "total_contents": len(content_df),
            "gender_dist": users_df["gender"].value_counts().to_dict(),
            "age_dist": users_df["age_range"].value_counts().to_dict(),
            "city_level_dist": users_df["fre_city_level"].value_counts().to_dict(),
            "category_dist": content_df["first_level_category_name"].value_counts().to_dict(),
        }

    # --- Internal loaders ---

    def _load_users_df(self) -> pd.DataFrame:
        if self._users_df is None:
            path = self.data_dir / "user_features_raw.csv"
            if not path.exists():
                raise FileNotFoundError(f"User features not found: {path}")
            self._users_df = pd.read_csv(path)
        return self._users_df

    def _load_content_df(self) -> pd.DataFrame:
        if self._content_df is None:
            path = self.data_dir / "kuairec_caption_category.csv"
            if not path.exists():
                raise FileNotFoundError(f"Caption category not found: {path}")
            self._content_df = pd.read_csv(path, engine="python", on_bad_lines="skip")
            # Filter out rows with no caption
            self._content_df = self._content_df[
                self._content_df["caption"].notna()
                & (self._content_df["caption"] != "")
                & (self._content_df["caption"] != "UNKNOWN")
            ]
        return self._content_df

    # --- Converters ---

    @staticmethod
    def _row_to_user(row) -> UserProfile | None:
        """Convert a DataFrame row to UserProfile."""
        try:
            # Gender
            gender_raw = str(row.get("gender", "")).strip()
            if gender_raw == "F":
                gender = Gender.FEMALE
            elif gender_raw == "M":
                gender = Gender.MALE
            else:
                gender = Gender.UNKNOWN

            # Age
            age_range = str(row.get("age_range", "unknown"))

            # Geography
            province = str(row.get("fre_province", "unknown"))
            city = str(row.get("fre_city", "unknown"))
            city_level = str(row.get("fre_city_level", "unknown"))

            # Device
            mod_price = row.get("mod_price", 0)
            try:
                mod_price = int(mod_price)
            except (ValueError, TypeError):
                mod_price = 1000
            if mod_price >= 3000:
                device_price = "high"
            elif mod_price >= 1000:
                device_price = "mid"
            else:
                device_price = "low"

            # Active degree
            active_raw = str(row.get("user_active_degree", "medium"))
            if "full" in active_raw or "high" in active_raw:
                active_degree = "high"
            elif "low" in active_raw:
                active_degree = "low"
            else:
                active_degree = "medium"

            # Interests — inferred from installed apps and activity
            interests = []
            if row.get("is_install_douyin") == 1:
                interests.append("短视频")
            if row.get("is_install_huoshan") == 1:
                interests.append("直播")
            if row.get("is_install_xigua") == 1:
                interests.append("长视频")
            if row.get("is_install_douyu") == 1 or row.get("is_install_huya") == 1:
                interests.append("游戏")
            if not interests:
                interests = ["综合"]

            return UserProfile(
                user_id=f"ku_{row['user_id']}",
                gender=gender,
                age_range=age_range,
                province=province,
                city=city,
                city_level=city_level,
                interests=interests,
                device_price=device_price,
                active_degree=active_degree,
            )
        except Exception:
            return None

    @staticmethod
    def _row_to_content(row) -> ContentItem | None:
        """Convert a DataFrame row to ContentItem."""
        try:
            # Parse topic tags
            topic_tags = []
            raw_tags = row.get("topic_tag", "[]")
            if pd.notna(raw_tags) and raw_tags not in ("[]", "", "UNKNOWN"):
                try:
                    parsed = ast.literal_eval(str(raw_tags))
                    if isinstance(parsed, list):
                        topic_tags = [str(t) for t in parsed[:5]]
                except (ValueError, SyntaxError):
                    topic_tags = []

            # Categories
            cat_l1 = str(row.get("first_level_category_name", "其他"))
            cat_l2 = str(row.get("second_level_category_name", "UNKNOWN"))
            cat_l3 = str(row.get("third_level_category_name", "UNKNOWN"))

            if cat_l1 in ("nan", "UNKNOWN", ""):
                cat_l1 = "其他"

            # Caption
            caption = str(row.get("caption", ""))
            if not caption or caption == "nan":
                return None

            return ContentItem(
                item_id=f"kv_{row['video_id']}",
                caption=caption[:200],  # Truncate very long captions
                topic_tags=topic_tags,
                category_l1=cat_l1,
                category_l2=cat_l2 if cat_l2 != "UNKNOWN" else cat_l1,
                category_l3=cat_l3 if cat_l3 != "UNKNOWN" else cat_l2,
            )
        except Exception:
            return None
