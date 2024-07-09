import requests
from taipy.gui import Gui, State, notify
from openai import OpenAI

# Point to the local server
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

context = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly."
## Conversation history
conversation = {
    "Conversation": ["Present Yourself", "Hi! I am LLaMag. How can I help you today?"]
}
current_user_message = ""
past_conversations = []
history = [
    {"role": "system", "content": "You are an intelligent assistant. You always provide well-reasoned answers that are both correct and helpful."},
    {"role": "user", "content": " "},
]

# Create a function that takes as input a string prompt which is the user message and returns a string which is the response from the LLM.
def query(state: State, prompt: str) -> str:
    completion = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B",
        messages=history,
        temperature=0.7,
        stream=True,
    )
    new_message = {"role": "assistant", "content": ""}
    response_text = ""
    for chunk in completion:
        if chunk.choices[0].delta.content:
            response_text += chunk.choices[0].delta.content
            new_message["content"] += chunk.choices[0].delta.content

    history.append(new_message)
    return response_text

def send_message(state: State) -> None:
    """
    Send the user's message to the API and update the conversation.

    Args:
        - state: The current state of the app.
    """
    # Add the user's message to the history
    history.append({"role": "user", "content": state.current_user_message})
    
    # Send the user's message to the API and get the response
    answer = query(state, state.current_user_message).replace("\n", "")
    
    # Update the conversation
    conv = state.conversation._dict.copy()
    conv["Conversation"] += [state.current_user_message, answer]
    state.conversation = conv
    
    # Clear the input field
    state.current_user_message = ""

def style_conv(state: State, idx: int, row: int) -> str:
    """
    Apply a style to the conversation table depending on the message's author.

    Args:
        - state: The current state of the app.
        - idx: The index of the message in the table.
        - row: The row of the message in the table.

    Returns:
        The style to apply to the message.
    """
    if idx is None:
        return None
    elif idx % 2 == 0:
        return "user_message"
    else:
        return "llamag_message"

# In Taipy, one way to define pages is to use Markdown strings. Here we use a table to display the conversation dictionary and an input so that the user can type their message. When the user presses enter, the send_message function is triggered.
page = """
<|layout|columns=300px 1|
<|part|class_name=sidebar|
# LLaMag **Chat**{: .color-primary} # {: .logo-text}
|>

<|part|class_name=p2 align-item-bottom table|
<|{conversation}|table|style=style_conv|show_all|>
<|{current_user_message}|input|label=Write your message here...|on_action=send_message|class_name=fullwidth|>
|>
|>
"""

if __name__ == "__main__":
    Gui(page).run(debug=True, dark_mode=True, use_reloader=True, title="LLaMag")
