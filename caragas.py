import panel as pn
pn.extension()
button = pn.widgets.Button(name='Click me', button_type='primary')
indicator = pn.indicators.LoadingSpinner(value=False, size=25)

def new_conversation(event):
    if not event:
        return
    
    print("Hello")

pn.bind(new_conversation, button, watch=True)

layout = pn.Column(button, indicator)

pn.serve(layout)