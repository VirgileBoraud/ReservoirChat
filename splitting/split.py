# Load the Q&A data
with open('doc/Q&A_format.md', 'r', encoding='utf-8') as file:
    data = file.read()

# Manually split the document based on headers
questions_answers = data.split("Question: ")

# Process the split data into a structured format
qa_pairs = []
for qa in questions_answers[1:]:  # Skipping the first empty split
    parts = qa.split("Answer: ")
    question = parts[0].strip()
    answer = parts[1].strip() if len(parts) > 1 else ""
    qa_pairs.append({'question': question, 'answer': answer})

print(qa_pairs)

# Write the structured data to a new file
with open('splitting/spliting_format.md', 'a', encoding='utf-8') as file:
    for pair in qa_pairs:
        file.write(f"{{question: \"{pair['question']}\", answer: \"{pair['answer']}\"}}\n")
