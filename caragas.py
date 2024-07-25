import panel as pn

pn.extension()

# Create a button
highlight_button = pn.widgets.Button(name='Highlight Me', button_type='primary')

# Function to highlight the button
def highlight(event):
    highlight_button.css_classes = ['highlight']

# Button to trigger the highlight
trigger_button = pn.widgets.Button(name='Trigger Highlight', button_type='success')
trigger_button.on_click(highlight)

# Define CSS for highlighting
highlight_css = """
<style>
.highlight {
    background-color: yellow !important;
    color: black !important;
    border-color: red !important;
}
</style>
"""

# Add the CSS to the document
pn.config.raw_css.append(highlight_css)

# Layout to display the buttons
layout = pn.Column(
    pn.pane.Markdown("# Button Highlight Example"),
    highlight_button,
    trigger_button
)

pn.serve(layout)
