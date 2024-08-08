
import panel as pn
from panel.chat import ChatInterface, ChatMessage

# Initialize Panel
pn.extension()

# Define a function to handle reaction clicks
def reaction_click_handler(event):
    print(f"Reaction clicked: {event.obj.reactions}")

# Create a ChatMessage with reactions
message = ChatMessage("Hello, how are you?",
    user="User",
    reactions=["favorite"],
    show_reaction_icons=False
)

# Watch the 'reactions' parameter for changes
#message.param.watch(reaction_click_handler, 'reactions')

# Create the ChatInterface and add the message
chat = ChatInterface(user="You")
chat.append(message)

# Serve the application
pn.serve(chat, title="Chat Interface", port=8080)
