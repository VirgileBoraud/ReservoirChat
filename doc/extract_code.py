import requests
import nbformat
import os

def download_notebook(github_url, save_path='notebook.ipynb'):
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

def create_markdown(code_cells, title='Extracted Code from Notebook'):
    md_content = f"# {title}\n\n"
    last_title = ""
    for i, (cell_title, code) in enumerate(code_cells):
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
    # Replace with the raw GitHub URL of the .ipynb file
    github_url = 'https://github.com/reservoirpy/reservoirpy/blob/master/tutorials/1-Getting_Started.ipynb'
    notebook_path = 'doc/notebook/1-Getting_Started.ipynb'
    markdown_path = 'doc/md/codes.md'

    # Download the notebook
    download_notebook(github_url, notebook_path)

    # Extract code cells with titles
    code_cells = extract_code_cells_with_titles(notebook_path)

    # Create Markdown content
    md_content = create_markdown(code_cells)

    # Save Markdown document
    save_markdown(md_content, markdown_path)

    # Clean up the downloaded notebook file
    if os.path.exists(notebook_path):
        os.remove(notebook_path)

if __name__ == "__main__":
    main()