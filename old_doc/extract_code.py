import requests
import nbformat
import os

def download_notebook(github_url, save_path):
    # Modify the URL to point to the raw content
    if not github_url.startswith("https://raw.githubusercontent.com/"):
        github_url = github_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    
    response = requests.get(github_url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded notebook from {github_url}")
    else:
        raise Exception(f"Failed to download notebook. Status code: {response.status_code}")

def extract_code_cells_with_titles(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)

    code_cells = []
    current_title = None
    for cell in notebook.cells:
        if cell.cell_type == 'markdown':
            lines = cell.source.split('\n')
            for line in lines:
                if line.startswith('#'):
                    current_title = line.strip('#').strip()
        elif cell.cell_type == 'code':
            if current_title:
                code_cells.append((current_title, cell.source))
            else:
                code_cells.append(("No Title", cell.source))
    return code_cells

def create_markdown(all_code_cells, title='Extracted Code from Notebooks'):
    md_content = f"# {title}\n\n"
    last_title = ""
    for i, (cell_title, code) in enumerate(all_code_cells):
        if cell_title != last_title:
            md_content += f"## {cell_title}\n"
            last_title = cell_title
        md_content += "```python\n"
        md_content += f"{code}\n"
        md_content += "```\n\n"
    return md_content

def save_markdown(md_content, md_path='codes.md'):
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"Saved Markdown document to {md_path}")

def main():
    github_urls = [
        'https://github.com/reservoirpy/reservoirpy/blob/master/tutorials/1-Getting_Started.ipynb',
        'https://github.com/reservoirpy/reservoirpy/blob/master/tutorials/2-Advanced_Features.ipynb',
        'https://github.com/reservoirpy/reservoirpy/blob/master/tutorials/3-General_Introduction_to_Reservoir_Computing.ipynb',
        'https://github.com/reservoirpy/reservoirpy/blob/master/tutorials/4-Understand_and_optimize_hyperparameters.ipynb',
        'https://github.com/reservoirpy/reservoirpy/blob/master/tutorials/5-Classification-with-RC.ipynb',
        'https://github.com/reservoirpy/reservoirpy/blob/master/tutorials/6-Interfacing_with_scikit-learn.ipynb',
    ]

    notebook_paths = ['doc/notebook/notebook_{}.ipynb'.format(i) for i in range(1, len(github_urls)+1)]
    markdown_path = 'doc/md/codes.md'

    all_code_cells = []

    # Download and process each notebook
    for i, github_url in enumerate(github_urls):
        notebook_path = notebook_paths[i]

        # Download the notebook
        download_notebook(github_url, notebook_path)

        # Extract code cells with titles
        code_cells = extract_code_cells_with_titles(notebook_path)
        all_code_cells.extend(code_cells)

    # Create Markdown content
    md_content = create_markdown(all_code_cells)

    # Save Markdown document
    save_markdown(md_content, markdown_path)

if __name__ == "__main__":
    main()