import panel as pn
import param

pn.extension()

# Define a class that will manage the buttons and track clicks
class ButtonTracker(param.Parameterized):
    # Define actions for each button
    button1 = param.Action(lambda x: x.param.trigger('button1'), label='Button 1')
    button2 = param.Action(lambda x: x.param.trigger('button2'), label='Button 2')
    button3 = param.Action(lambda x: x.param.trigger('button3'), label='Button 3')

    # Initialize the class
    def __init__(self, **params):
        super().__init__(**params)
        # Create a layout to hold the buttons
        self.layout = pn.Column()
        # Map button names to their corresponding Panel Button widgets
        self.button_map = {
            'button1': pn.widgets.Button(name='Button 1'),
            'button2': pn.widgets.Button(name='Button 2'),
            'button3': pn.widgets.Button(name='Button 3')
        }
        # Add each button to the layout and bind click events to the button_click method
        for name, button in self.button_map.items():
            button.on_click(lambda event, name=name: self.button_click(name))
            self.layout.append(button)
        # Watch for the button actions and bind them to the button_click method
        self.param.watch(self.button_click, 'button1')
        self.param.watch(self.button_click, 'button2')
        self.param.watch(self.button_click, 'button3')

    # Method to handle button clicks
    def button_click(self, name):
        # Get the position of the clicked button in the layout
        position = self.layout.objects.index(self.button_map[name])
        # Print the name and position of the clicked button
        print(f"Button {self.button_map[name].name} was clicked at position {position}")

# Create an instance of the class
button_tracker = ButtonTracker()

# Serve the layout so it can be viewed in a web browser
pn.serve(button_tracker.layout)
