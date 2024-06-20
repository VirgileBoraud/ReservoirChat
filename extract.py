import markdown
from bs4 import BeautifulSoup

# Function to parse the markdown file
def parse_markdown(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    html = markdown.markdown(text)
    soup = BeautifulSoup(html, 'html.parser')
    return soup

# Function to extract Q&A from the parsed markdown
def extract_qa(soup):
    qa_pairs = []
    current_question = None
    current_answer = []
    
    for element in soup.find_all(['h2', 'p', 'ul', 'li']):
        if element.name == 'h2' and current_question:
            qa_pairs.append((current_question, ' '.join(current_answer)))
            current_question = None
            current_answer = []

        if element.name == 'p' and element.text.startswith('Q:'):
            if current_question:
                qa_pairs.append((current_question, ' '.join(current_answer)))
            current_question = element.text[3:].strip()
            current_answer = []
        elif element.name == 'p' and element.text.startswith('A:'):
            current_answer.append(element.text[3:].strip())
        elif element.name == 'li':
            current_answer.append(element.text.strip())

    if current_question:
        qa_pairs.append((current_question, ' '.join(current_answer)))

    return qa_pairs

# Parse the markdown file
soup = parse_markdown('Q&A.md')

# Extract Q&A pairs from the parsed content
qa_pairs = extract_qa(soup)


writing = ""
for question, answer in qa_pairs:
    writing += "Q: " + question + answer + "\n"
 
with open('Q&A_format.md', 'w', encoding='utf-8') as file:       
    file.write(writing)