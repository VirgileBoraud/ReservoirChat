import time
from graphrag.query.cli import run_local_search

def ask(questions_file, output_file, temperature):
    """
    Reads questions from a markdown file, queries an LLM for responses, 
    and saves the responses in a specified markdown file.

    Parameters:
        questions_file (str): Path to the markdown file containing questions.
        output_file (str): Path to the markdown file where responses will be saved.
        temperature (float): The temperature to use for the LLM responses (if needed).
    """

    # Local LLM interaction function using the streaming method
    def get_response(user_message, history):
        print('--------------Get_Response History-----------------')
        print(history)
        print('---------------------------------------------------')
        if history != []:
            if not isinstance(history[0], dict):
                history = history[0]
                print('--------------Get_Response History Check-----------------')
                print(history)
                print('---------------------------------------------------')

        message = run_local_search('ragtest',
                                   'ragtest/output/basic/artifacts',
                                   'ragtest',
                                   0,
                                   'This is a response',
                                   False,
                                   user_message,
                                   history)
        response_text = ""
        for chunk in message:
            response_text += chunk
            # Collecting chunks in response_text
            time.sleep(0.01)

        # Return the final response text after streaming
        return response_text

    # Read the questions from the markdown file
    with open(questions_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Initialize an empty list to store the formatted responses
    responses = []
    current_level = None

    # Process each line and generate responses for questions
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith("**") and stripped_line.endswith("**"):  # Detects level (e.g., **Beginner**, **Intermediate**, **Advanced**)
            current_level = stripped_line  # Maintain the bold formatting as is
            responses.append(f"{current_level}\n\n")
        elif line.startswith("- "):  # Identifies a question line
            question = line[2:].strip()

            # Get three responses with conversation history support
            response_texts = []
            for _ in range(3):
                # Collect the complete response (after streaming)
                complete_response = get_response(question, [])
                response_texts.append(complete_response)

            # Append the question and responses side by side in markdown format under the current level
            responses.append(f"## Question: {question}\n### Responses:\n")
            for i, response_text in enumerate(response_texts, 1):
                responses.append(f"**Response {i}:**\n{response_text}\n\n")

    # Write the responses to the output markdown file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.writelines(responses)

    print(f"Responses have been written to {output_file}")


# Example usage
input_files = ['./old_doc/md/questions.md', './old_doc/md/questions_specified.md']
temperatures = [0.7]

for questions_file in input_files:
    for temp in temperatures:
        output_file = f"question_and_responses_reservoirchat_{questions_file.split('/')[-1].replace('.md', '')}_{temp}.md"
        ask(questions_file, output_file, temp)
