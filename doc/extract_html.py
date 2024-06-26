import requests
from bs4 import BeautifulSoup

# Function to extract titles, text, math elements, libraries, and references
def extract_content(soup):
    content = ''
    # Extract titles, paragraphs, and math elements
    for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'math']):
        if element.name.startswith('h'):
            content += f"# {element.get_text().strip()}\n\n"
        elif element.name == 'p':
            content += f"{element.get_text().strip()}\n\n"
        elif element.name == 'math':
            content += f"${element.get_text().strip()}$\n\n"

    # Extract libraries
    libraries_section = soup.find(id="Libraries")
    if libraries_section:
        content += f"## {libraries_section.get_text().strip()}\n\n"
        for sibling in libraries_section.find_next_siblings():
            if sibling.name == 'h2':
                break
            if sibling.name == 'p':
                content += f"{sibling.get_text().strip()}\n\n"
            elif sibling.name == 'ul':
                for li in sibling.find_all('li'):
                    content += f"- {li.get_text().strip()}\n\n"

    # Extract references
    references_section = soup.find(id="References")
    if references_section:
        content += f"## {references_section.get_text().strip()}\n\n"
        for sibling in references_section.find_next_siblings():
            if sibling.name == 'h2':
                break
            if sibling.name == 'p':
                content += f"{sibling.get_text().strip()}\n\n"
            elif sibling.name == 'ol':
                for li in sibling.find_all('li'):
                    content += f"- {li.get_text().strip()}\n\n"

    return content

# URL of the Wikipedia page
url = "http://www.scholarpedia.org/article/Recurrent_neural_networks"

# Send a GET request to the webpage
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Extract the relevant content
md_content = extract_content(soup)

# Save the content to a markdown file
with open('doc/md/RNN_Scholarpedia.md', 'w', encoding='utf-8') as file:
    file.write(md_content)

print("Markdown file has been created.")