import json
from smolagents import tool, LiteLLMModel, CodeAgent

# Load real outfit data from Polyvore
with open("train.json", "r") as file:
    outfits = json.load(file)

# Load information about each clothing item
with open("polyvore_item_metadata.json", "r") as file:
    item_metadata = json.load(file)


# ---- Pure logic functions (testable without smolagents or the agent) ----

def _get_outfit(index, outfits):
    return outfits[index]

def _get_item_info(item_id, item_metadata):
    return item_metadata.get(item_id, {"error": "Item not found"})

def _get_categories(outfit, item_metadata):
    """Return the list of semantic categories present in one outfit."""
    categories = []
    for item in outfit["items"]:
        item_id = item["item_id"]
        item_info = item_metadata.get(item_id, {})
        semantic_category = item_info.get("semantic_category")
        if semantic_category:
            categories.append(semantic_category)
    return categories

def _search_by_category(category, outfits, item_metadata, limit=5):
    matches = []
    for outfit in outfits:
        if category in _get_categories(outfit, item_metadata):
            matches.append(outfit)
        if len(matches) >= limit:
            break
    return matches

def _search_by_categories(categories, outfits, item_metadata, limit=5):
    matches = []
    for outfit in outfits:
        outfit_categories = _get_categories(outfit, item_metadata)
        if all(category in outfit_categories for category in categories):
            matches.append({
                "set_id": outfit["set_id"],
                "categories": outfit_categories
            })
        if len(matches) >= limit:
            break
    return matches


# ---- Tools (thin wrappers around the logic above) ----

@tool
def get_outfit(index: int) -> dict:
    """Get one Polyvore outfit by its position in the dataset.

    Args:
        index: The position number of the outfit in the dataset.
    """
    return _get_outfit(index, outfits)

@tool
def get_item_info(item_id: str) -> dict:
    """Get information about one clothing item by its ID.

    Args:
        item_id: The ID of the clothing item.
    """
    return _get_item_info(item_id, item_metadata)

@tool
def search_outfits_by_category(category: str, limit: int = 5) -> list:
    """Search Polyvore outfits that contain a clothing category.

    Args:
        category: The semantic clothing category to search for, such as tops, shoes, or bags.
        limit: The maximum number of matching outfits to return.
    """
    return _search_by_category(category, outfits, item_metadata, limit)

@tool
def search_outfits_by_categories(categories: list[str], limit: int = 5) -> list:
    """Search Polyvore outfits that contain all requested clothing categories.

    Returns a list of dictionaries. Each dictionary contains:
    - "set_id": the outfit's ID
    - "categories": a list of clothing categories in the outfit

    Args:
        categories: A list of clothing categories that must all appear in the outfit.
        limit: The maximum number of matching outfits to return.
    """
    return _search_by_categories(categories, outfits, item_metadata, limit)


# Model: connects your program to the Qwen2 model running locally through Ollama
model = LiteLLMModel(
    model_id="ollama_chat/qwen2:7b",
    api_base="http://127.0.0.1:11434",
    num_ctx=8192,
)

# Agent
agent = CodeAgent(
    tools=[
        get_outfit,
        get_item_info,
        search_outfits_by_category,
        search_outfits_by_categories
    ],
    model=model,
)

# Version 6 Test B
if __name__ == "__main__":
    agent.run(
        "Find 3 Polyvore outfits containing both tops and shoes. "
        "For each outfit, give me its set_id and clothing categories."
    )
