import json

# Load the provided Jupyter Notebook file
with open('doc/notebook/Edge_of_stability_Ceni_Gallicchio_2023.ipynb', 'r', encoding='utf-8') as file:
    notebook_content = file.read()

# Parse the Jupyter Notebook content
notebook_data = json.loads(notebook_content)

# Initialize markdown content
markdown_content = ""

# Extract cells from the notebook
for cell in notebook_data['cells']:
    if cell['cell_type'] == 'markdown':
        markdown_content += '\n'.join(cell['source']) + '\n\n'
    elif cell['cell_type'] == 'code':
        markdown_content += '```python\n' + '\n'.join(cell['source']) + '\n```\n\n'

# Save the extracted content to a markdown file
markdown_file_path = 'doc/md/Edge_of_stability_Ceni_Gallicchio_2023.md'
with open(markdown_file_path, 'w', encoding='utf-8') as markdown_file:
    markdown_file.write(markdown_content)

markdown_file_path
