import pytest
from closet_agent import (
    _get_outfit,
    _get_item_info,
    _get_categories,
    _search_by_category,
    _search_by_categories,
)


@pytest.fixture
def sample_metadata():
    return {
        "item1": {"semantic_category": "tops"},
        "item2": {"semantic_category": "shoes"},
        "item3": {"semantic_category": "bags"},
        "item4": {},  # no semantic_category on purpose
    }


@pytest.fixture
def sample_outfits():
    return [
        {
            "set_id": "outfit_0",
            "items": [{"item_id": "item1"}, {"item_id": "item2"}],
        },
        {
            "set_id": "outfit_1",
            "items": [{"item_id": "item3"}],
        },
        {
            "set_id": "outfit_2",
            "items": [{"item_id": "item1"}, {"item_id": "item3"}],
        },
    ]


# --- get_outfit ---

def test_get_outfit_returns_correct_outfit(sample_outfits):
    result = _get_outfit(1, sample_outfits)
    assert result["set_id"] == "outfit_1"


# --- get_item_info ---

def test_get_item_info_returns_known_item(sample_metadata):
    result = _get_item_info("item1", sample_metadata)
    assert result == {"semantic_category": "tops"}

def test_get_item_info_missing_item_returns_error(sample_metadata):
    result = _get_item_info("nonexistent", sample_metadata)
    assert result == {"error": "Item not found"}


# --- get_categories ---

def test_get_categories_returns_expected_list(sample_outfits, sample_metadata):
    result = _get_categories(sample_outfits[0], sample_metadata)
    assert result == ["tops", "shoes"]

def test_get_categories_skips_items_with_no_category(sample_metadata):
    outfit = {"items": [{"item_id": "item4"}]}
    result = _get_categories(outfit, sample_metadata)
    assert result == []


# --- search_by_category ---

def test_search_by_category_finds_matching_outfits(sample_outfits, sample_metadata):
    result = _search_by_category("tops", sample_outfits, sample_metadata)
    set_ids = [outfit["set_id"] for outfit in result]
    assert set_ids == ["outfit_0", "outfit_2"]

def test_search_by_category_respects_limit(sample_outfits, sample_metadata):
    result = _search_by_category("tops", sample_outfits, sample_metadata, limit=1)
    assert len(result) == 1

def test_search_by_category_no_match_returns_empty(sample_outfits, sample_metadata):
    result = _search_by_category("dresses", sample_outfits, sample_metadata)
    assert result == []


# --- search_by_categories ---

def test_search_by_categories_requires_all_categories(sample_outfits, sample_metadata):
    result = _search_by_categories(["tops", "shoes"], sample_outfits, sample_metadata)
    assert len(result) == 1
    assert result[0]["set_id"] == "outfit_0"

def test_search_by_categories_returns_correct_structure(sample_outfits, sample_metadata):
    result = _search_by_categories(["tops"], sample_outfits, sample_metadata)
    assert "set_id" in result[0]
    assert "categories" in result[0]

def test_search_by_categories_no_match_returns_empty(sample_outfits, sample_metadata):
    result = _search_by_categories(["dresses", "hats"], sample_outfits, sample_metadata)
    assert result == []
