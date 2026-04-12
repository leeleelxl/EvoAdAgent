"""Tests for simulation scenarios."""

import pytest

from src.models import ContentItem, Gender, UserProfile
from src.simulation.scenarios import (
    create_food_scenario,
    create_full_scenario,
    create_pet_scenario,
    create_sample_contents,
    create_sample_users,
)


# ============================================================
# create_sample_users
# ============================================================


class TestCreateSampleUsers:
    def test_default_returns_20_users(self):
        users = create_sample_users()
        assert len(users) == 20

    def test_custom_count(self):
        users = create_sample_users(n=5)
        assert len(users) == 5

    def test_returns_user_profile_instances(self):
        users = create_sample_users(3)
        for user in users:
            assert isinstance(user, UserProfile)

    def test_each_user_has_unique_id(self):
        users = create_sample_users()
        ids = [u.user_id for u in users]
        assert len(ids) == len(set(ids))

    def test_each_user_has_complete_attributes(self):
        users = create_sample_users()
        for user in users:
            assert user.user_id, "user_id should not be empty"
            assert isinstance(user.gender, Gender)
            assert user.age_range, "age_range should not be empty"
            assert user.province, "province should not be empty"
            assert user.city, "city should not be empty"
            assert user.city_level, "city_level should not be empty"
            assert isinstance(user.interests, list)
            assert len(user.interests) > 0, "every sample user should have at least one interest"
            assert user.device_price in ("low", "mid", "high")
            assert user.active_degree in ("low", "medium", "high")

    def test_gender_variety(self):
        users = create_sample_users()
        genders = {u.gender for u in users}
        assert Gender.MALE in genders
        assert Gender.FEMALE in genders

    def test_city_level_variety(self):
        users = create_sample_users()
        levels = {u.city_level for u in users}
        assert len(levels) >= 2, "should have at least 2 different city levels"

    def test_n_larger_than_templates_capped(self):
        users = create_sample_users(n=100)
        assert len(users) == 20  # only 20 templates defined


# ============================================================
# create_sample_contents
# ============================================================


class TestCreateSampleContents:
    def test_returns_non_empty_list(self):
        contents = create_sample_contents()
        assert len(contents) > 0

    def test_returns_content_item_instances(self):
        contents = create_sample_contents()
        for c in contents:
            assert isinstance(c, ContentItem)

    def test_each_content_has_unique_id(self):
        contents = create_sample_contents()
        ids = [c.item_id for c in contents]
        assert len(ids) == len(set(ids))

    def test_each_content_has_complete_attributes(self):
        contents = create_sample_contents()
        for c in contents:
            assert c.item_id, "item_id should not be empty"
            assert c.caption, "caption should not be empty"
            assert isinstance(c.topic_tags, list)
            assert len(c.topic_tags) > 0, "each content should have at least one tag"
            assert c.category_l1, "category_l1 should not be empty"
            assert c.category_l2, "category_l2 should not be empty"
            assert c.category_l3, "category_l3 should not be empty"
            assert c.duration_seconds > 0, "duration should be positive"

    def test_multiple_categories_present(self):
        contents = create_sample_contents()
        categories = {c.category_l1 for c in contents}
        assert len(categories) >= 3, "should have at least 3 different L1 categories"

    def test_known_categories_present(self):
        contents = create_sample_contents()
        categories = {c.category_l1 for c in contents}
        assert "宠物" in categories
        assert "美食" in categories
        assert "搞笑" in categories


# ============================================================
# create_pet_scenario
# ============================================================


class TestCreatePetScenario:
    def test_returns_users_and_contents(self):
        users, contents = create_pet_scenario()
        assert isinstance(users, list)
        assert isinstance(contents, list)

    def test_all_contents_are_pet_category(self):
        _, contents = create_pet_scenario()
        for c in contents:
            assert c.category_l1 == "宠物", f"expected 宠物 but got {c.category_l1}"

    def test_returns_at_least_one_pet_content(self):
        _, contents = create_pet_scenario()
        assert len(contents) > 0

    def test_users_are_full_set(self):
        users, _ = create_pet_scenario()
        assert len(users) == 20


# ============================================================
# create_food_scenario
# ============================================================


class TestCreateFoodScenario:
    def test_returns_users_and_contents(self):
        users, contents = create_food_scenario()
        assert isinstance(users, list)
        assert isinstance(contents, list)

    def test_all_contents_are_food_category(self):
        _, contents = create_food_scenario()
        for c in contents:
            assert c.category_l1 == "美食", f"expected 美食 but got {c.category_l1}"

    def test_returns_at_least_one_food_content(self):
        _, contents = create_food_scenario()
        assert len(contents) > 0

    def test_users_are_full_set(self):
        users, _ = create_food_scenario()
        assert len(users) == 20


# ============================================================
# create_full_scenario
# ============================================================


class TestCreateFullScenario:
    def test_returns_all_users_and_contents(self):
        users, contents = create_full_scenario()
        assert len(users) == 20
        assert len(contents) == len(create_sample_contents())

    def test_full_scenario_has_more_content_than_pet(self):
        _, full_contents = create_full_scenario()
        _, pet_contents = create_pet_scenario()
        assert len(full_contents) > len(pet_contents)

    def test_full_scenario_has_more_content_than_food(self):
        _, full_contents = create_full_scenario()
        _, food_contents = create_food_scenario()
        assert len(full_contents) > len(food_contents)
