import panel as pn
import param

# Function to create the app
def create_app():
    class UserState(param.Parameterized):
        name = param.String(default='')

        @param.depends('name', watch=True)
        def view(self):
            return pn.pane.Markdown(f"Hello, {self.name}!")

    user_state = UserState()
    name_input = pn.widgets.TextInput(name='Name')
    name_input.param.watch(lambda event: setattr(user_state, 'name', event.new), 'value')

    return pn.Column(name_input, user_state.view)

# Serve the app
pn.serve(create_app)
