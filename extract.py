import markdown
import json
import re

# Function to extract Q&A from the markdown
def extract_qa(md):
    qa_pairs = []
    current_question = None
    current_answer = []

    lines = md.split('\n')
    for line in lines:
        question_match = re.match(r'^##\s*(.*)', line)
        q_match = re.match(r'^Q:\s*(.*)', line)
        a_match = re.match(r'^A:\s*(.*)', line)
        li_match = re.match(r'^-\s*(.*)', line)

        if question_match and current_question:
            qa_pairs.append((current_question, ' '.join(current_answer)))
            current_question = None
            current_answer = []

        if q_match:
            if current_question:
                qa_pairs.append((current_question, ' '.join(current_answer)))
            current_question = q_match.group(1).strip()
            current_answer = []
        elif a_match:
            current_answer.append(a_match.group(1).strip())
        elif li_match:
            current_answer.append(li_match.group(1).strip())

    if current_question:
        qa_pairs.append((current_question, ' '.join(current_answer)))

    return qa_pairs

# Reading the markdown file content
with open('Q&A.md', 'r', encoding='utf-8') as file:
    QA = file.read()

qa_pairs = extract_qa(QA)

format = ""

for question, answer in qa_pairs:
    format += "Question: " + question + "\n" + "Answer: " + answer + "\n"

with open('Q&A_format.md', 'w', encoding='utf-8') as file:       
    file.write(format)

listing = [{"Question": question, "Answer": answer} for question, answer in qa_pairs]

with open('Q&A_list.json', 'w', encoding='utf-8') as file:
    json.dump(listing, file, ensure_ascii=False, indent=4)