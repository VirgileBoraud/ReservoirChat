import json
import re

# Function to extract Q&A from the markdown
def extract_qa(md):
    qa_pairs = []
    current_question = None
    current_answer = []
    collecting_answer = False

    lines = md.split('\n')
    for line in lines:
        # Checks for commentary or section dividers
        if re.match(r'^#|^---', line):
            if current_question is not None:
                qa_pairs.append((current_question, ' '.join(current_answer)))
                current_question = None
                current_answer = []
                collecting_answer = False
            continue

        # Matches for 'Q:' and 'A:'
        q_match = re.match(r'^Q:\s*(.*)', line)
        a_match = re.match(r'^A:\s*(.*)', line)

        if q_match:
            # Save the current Q&A pair if it exists
            if current_question is not None:
                qa_pairs.append((current_question, ' '.join(current_answer)))
            current_question = q_match.group(1).strip()
            current_answer = []
            collecting_answer = False
        elif a_match:
            collecting_answer = True
            current_answer.append(a_match.group(1).strip())
        elif collecting_answer:
            # Collect lines as part of the answer until the next 'Q:' or commentary
            current_answer.append(line.strip())

    # Append the last Q&A pair if it hasn't been added yet
    if current_question:
        qa_pairs.append((current_question, ' '.join(current_answer)))

    return qa_pairs

# Reading the markdown file content
with open('doc/Q&A.md', 'r', encoding='utf-8') as file:
    QA = file.read()

qa_pairs = extract_qa(QA)

# Formatting for Markdown output
format = ""
for question, answer in qa_pairs:
    format += "Question: " + question + "\n" + "Answer: " + answer + "\n"

with open('doc/Q&A_format.md', 'w', encoding='utf-8') as file:
    file.write(format)

# Creating a JSON listing
listing = [{"Question": question, "Answer": answer} for question, answer in qa_pairs]
with open('doc/Q&A_list.json', 'w', encoding='utf-8') as file:
    json.dump(listing, file, ensure_ascii=False, indent=4)