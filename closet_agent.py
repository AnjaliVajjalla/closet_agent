import json
from smolagents import tool, LiteLLMModel, CodeAgent

# Load real outfit data from Polyvore
with open("train.json", "r") as file:
    outfits = json.load(file)

# Load information about each clothing item
with open("polyvore_item_metadata.json", "r") as file:
    item_metadata = json.load(file)

# Temporary tests so we can inspect the data
print(outfits[0])
print(type(item_metadata))
print(next(iter(item_metadata.items())))

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
    tools=[get_outfit, get_item_info],
    # gives the agent permission to use that tool
    model=model,
    # tells the agent to use the model you created above as its brain
)

# Give the agent a Version 2 task
agent.run(
    "Look at outfit number 0 from the Polyvore dataset. "
    "Use the 'items' list to get each item_id. "
    "Then use get_item_info for each item and use the 'semantic_category' field "
    "to tell me what kinds of clothing are in the outfit."
)