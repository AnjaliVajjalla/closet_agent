import json
from smolagents import tool, LiteLLMModel, CodeAgent

# Load real outfit data from Polyvore
with open("train.json", "r") as file:
    outfits = json.load(file)

# Load information about each clothing item
with open("polyvore_item_metadata.json", "r") as file:
    item_metadata = json.load(file)

# Temporary tests so we can inspect the data
# print(outfits[0])
# print(type(item_metadata))
# print(next(iter(item_metadata.items())))

# Tool 1: get one outfit from the dataset
@tool
def get_outfit(index: int) -> dict:
    """Get one Polyvore outfit by its position in the dataset.

    Args:
        index: The position number of the outfit in the dataset.
    """
    return outfits[index]

# Tool 2: get information about one clothing item
@tool
def get_item_info(item_id: str) -> dict:
    """Get information about one clothing item by its ID.

    Args:
        item_id: The ID of the clothing item.
    """
    return item_metadata.get(item_id, {"error": "Item not found"})

# Tool 3: search outfits by clothing category
@tool
def search_outfits_by_category(category: str, limit: int = 5) -> list:
    """Search Polyvore outfits that contain a clothing category.

            Args:
                category: The semantic clothing category to search for, such as tops, shoes, or bags.
                limit: The maximum number of matching outfits to return.
    """
    # default number is 5 for limit if not specified 
    matches = []
    # create an empty list called matches to gradually put matching fits in this list

    # go through each outfit in the outfits list
    for outfit in outfits: 

        # for each fit, create a new empty list to collect its clothing categories
        outfit_categories = []

        # Loops through each clothing item inside fit and gets its semantic category from the item_metadata dictionary
        for item in outfit ["items"]:
            item_id = item["item_id"]
            item_info = item_metadata.get(item_id, {})
            # gets the useful categories like "tops", "shpes", "bags", etc.
            semantic_category = item_info.get("semantic_category")

            # if category exist, add it to the outfit_categories list
            if semantic_category:
                outfit_categories.append(semantic_category)

        # if category is present in the outfit, saves outfit to the matches list
        if category in outfit_categories:
            matches.append(outfit)

        # If we already found enough fits, stops the loop
        if len(matches) >= limit:
            break
    return matches

# Tool 4: search outfits by multiple clothing categories 
@tool
def search_outfits_by_categories(categories: list[str], limit: int = 5) -> list:
    """Search Polyvore outfits that contain all requested clothing categories.

Args:
    categories: A list of clothing categories that must all appear in the outfit.
    limit: The maximum number of matching outfits to return.
"""
    matches = []

    for outfit in outfits:
        outfit_categories = []

        for item in outfit["items"]:
            item_id = item["item_id"]
            item_info = item_metadata.get(item_id, {})
            semantic_category = item_info.get("semantic_category")

            if semantic_category:
                outfit_categories.append(semantic_category)

        # checks if all requested categories are present in the outfit
        if all(category in outfit_categories for category in categories):
            matches.append(outfit)

        if len(matches) >= limit:
            break
    return [outfit["set_id"] for outfit in matches]

# Model: connectes your program to the Qwen2 model running locally through Ollama
model = LiteLLMModel(
# creating something and storing it in a variable called: model
    model_id="ollama_chat/qwen2:7b",
    # which AI model to use: accessing it through ollama, and we're using the Qwen2 7B model
    api_base="http://127.0.0.1:11434",
    # where the model/server is
    num_ctx=8192,
    # context window size: how many tokens of context the model can work with
)

# Agent: 
agent = CodeAgent(
    tools=[
    get_outfit,
    get_item_info,
    search_outfits_by_category,
    search_outfits_by_categories
],
    # gives the agent permission to use that tool
    model=model,
    # tells the agent to use the model you created above as its brain
)

# Test Tasks: 
# Version 1 Test
# agent.run(
#     "Look at outfit number 0 from the Polyvore dataset and tell me how many items it contains."
# )

# Version 2 Test
# agent.run(
#     "Look at outfit number 0 from the Polyvore dataset. "
#     "Use the 'items' list to get each item_id. "
#     "Then use get_item_info for each item and use the 'semantic_category' field "
#     "to tell me what kinds of clothing are in the outfit."
# )

# Version 3 Test
# agent.run(
#     "Find 3 Polyvore outfits that contain shoes. "
#     "Use search_outfits_by_category. "
#     "The tool returns a list of outfit dictionaries. "
#     "Loop directly through that list and return the set_id from each outfit."
# )

# Version 4 Test
# agent.run(
#     "Find 3 Polyvore outfits that contain both tops and shoes. "
#     "Use search_outfits_by_categories and return the set_id of each matching outfit."
# )

# Version 5 Test A
# agent.run(
#     "Find 3 Polyvore outfits that contain shoes and return the set_id of each matching outfit."
# )

# Version 5 Test B
agent.run(
    "Find 3 Polyvore outfits that contain both tops and shoes and return the set_id of each matching outfit."
)