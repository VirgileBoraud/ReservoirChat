import panel as pn
pn.extension()

chat_input = pn.chat.ChatAreaInput(placeholder='Type a message...')

chat_feed = pn.chat.ChatFeed()

def handle_message(event):
    chat_feed.append(pn.chat.ChatMessage(text=event.new, name='User'))
    chat_input.value = ''

chat_input.param.watch(handle_message, 'value')

chat_interface = pn.Column(chat_feed, chat_input)
chat_interface.show()
