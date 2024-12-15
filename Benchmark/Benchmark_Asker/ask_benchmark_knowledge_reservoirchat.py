import time
from graphrag.query.cli import run_local_search

def ask(questions_file, output_file, temperature):
    """
    Reads questions from a markdown file, queries ReservoirChat for responses, 
    and saves the responses in a specified markdown file. Each question gets three responses in markdown format.

    Parameters:
        questions_file (str): Path to the markdown file containing questions.
        output_file (str): Path to the markdown file where responses will be saved.
        temperature (float): The temperature to use for the LLM responses (if needed).
    """

    # Function to get a response from ReservoirChat using streaming
    def get_response(question_block, history):
        """
        Uses ReservoirChat to fetch the response by streaming chunks from the model.
        
        Parameters:
            question_block (str): The block of text containing the question and answer options.
            history (list): The conversation history, passed as empty in this case.
        
        Returns:
            str: The accumulated response text from ReservoirChat.
        """
        # The prompt to instruct the model to answer with only the letter A, B, C, or D.
        system_prompt = (
            "You will be given a question with four answer options labeled A, B, C, and D. Please respond with only the letter (A, B, C, or D) that is the correct answer."
        )
        
        message = run_local_search(
            'ragtest',  # Replace with your actual task name
            'ragtest/output/basic/artifacts',  # Adjust the path according to your setup
            'ragtest',  # Replace as needed
            0,  # Adjust temperature or other parameters if necessary
            system_prompt,
            False,
            question_block,
            history
        )

        # Collect response chunks
        response_text = ""
        for chunk in message:
            response_text += chunk
            time.sleep(0.01)  # Simulate streaming delay

        return response_text.strip()

    # Read the questions from the markdown file
    with open(questions_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Initialize an empty list to store the formatted responses
    responses = []
    current_level = None
    question_block = ""

    # Process each line and generate responses for questions
    for line in lines:
        stripped_line = line.strip()

        # Detect levels like **Beginner**, **Intermediate**, **Advanced**
        if stripped_line.startswith("**") and stripped_line.endswith("**"):
            current_level = stripped_line
            responses.append(f"{current_level}\n\n")
        elif stripped_line == "---":  # End of question block marker
            if question_block:
                # Get three responses for the question block
                response_texts = [get_response(question_block, []) for _ in range(3)]

                # Append question and its responses
                responses.append(f"## Question:\n{question_block.strip()}\n### Responses:\n")
                for i, response_text in enumerate(response_texts, 1):
                    responses.append(f"**Response {i}:**\n{response_text}\n\n")
                
                question_block = ""  # Reset for the next question block
        else:
            question_block += line + "\n"  # Continue building the question block

    # Process the last question block if there is no final `---`
    if question_block:
        response_texts = [get_response(question_block, []) for _ in range(3)]
        responses.append(f"## Question:\n{question_block.strip()}\n### Responses:\n")
        for i, response_text in enumerate(response_texts, 1):
            responses.append(f"**Response {i}:**\n{response_text}\n\n")

    # Write the formatted responses to the output markdown file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.writelines(responses)

    print(f"Responses have been written to {output_file}")

# Example usage
input_files = ['./old_doc/md/benchmark_knowledge.md']  # Specify your input markdown file here
temperatures = [0.1]  # Temperature setting for diversity

for questions_file in input_files:
    for temp in temperatures:
        output_file = f"question_and_responses_reservoirchat_{questions_file.split('/')[-1].replace('.md', '')}_{temp}.md"
        ask(questions_file, output_file, temp)
