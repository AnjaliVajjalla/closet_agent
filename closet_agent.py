import json
from smolagents import tool, LiteLLMModel, CodeAgent

# This is my closet data
# closet = [
#     "black fitted top",
#     "pink tank top",
#     "white blouse",
#     "blue jeans",
#     "black mini skirt",
#     "black trousers",
#     "white sneakers",
#     "black heels"
# ]

# Load real outfit data from Polyvore
with open("train.json", "r") as file:
    outfits = json.load(file)

# Test: show the first outfit
print(outfits[0])

# Tool: lets the agent access the closet

@tool
def get_outfit(index: int) -> dict:
    """Get one Polyvore outfit by its position in the dataset.

    Args:
        index: The position number of the outfit in the dataset.
    """
    return outfits[index]

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
    tools=[get_outfit],
    # gives the agent permission to use that tool
    model=model,
    # tells the agent to use the model you created above as its brain
)

# Give Agent a Task: 
agent.run(
    "Look at outfit number 0 from the Polyvore dataset and tell me how many items it contains."
)



