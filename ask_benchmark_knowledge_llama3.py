import requests
import json

def ask(questions_file, output_file, model_url, model_name, api_key, temperature):
    """
    Reads questions from a markdown file, queries an LLM for responses, 
    and saves the responses in a specified markdown file. Each response is a single letter.
    
    Parameters:
        questions_file (str): Path to the markdown file containing questions.
        output_file (str): Path to the markdown file where responses will be saved.
        model_url (str): The URL of the LLM API endpoint.
        model_name (str): The model name used in the LLM API.
        api_key (str): The API key for accessing the LLM.
        temperature (float): The temperature to use for the LLM responses.
    """

    # Function to get a response from the LLM
    def get_response(question_block):
        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'}
        payload = {
            "model": model_name,
            "messages": [{"role": "system", "content": "You will be given a question with four answer options labeled A, B, C, and D. Please respond using the library reservoirPy with only the letter (A, B, C, or D) that is the correct answer."},
                         {"role": "user", "content": question_block}],
            "temperature": temperature
        }

        try:
            response = requests.post(model_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()  # Raises an HTTPError if the response code was unsuccessful

            # Try to parse the expected content
            data = response.json()

            # Check if 'choices' and 'message' keys exist in the response
            if 'choices' in data and data['choices']:
                # Strip and ensure the response contains only a valid letter
                answer = data['choices'][0]['message'].get('content', 'No content found').strip().upper()
                if answer in ['A', 'B', 'C', 'D']:
                    return answer
                else:
                    return "Invalid response: " + answer
            else:
                return "Unexpected response structure: " + json.dumps(data)

        except requests.exceptions.RequestException as e:
            print(f"HTTP Request failed: {e}")
            return "Error: Could not retrieve the response due to a request failure."

    # Read the questions from the markdown file
    with open(questions_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Initialize variables to store responses
    responses = []
    current_level = None
    question_block = ""
    formatted_responses = []

    # Process each line and generate responses for questions
    for line in lines:
        stripped_line = line.strip()

        if stripped_line.startswith("**") and stripped_line.endswith("**"):  # Detects level (e.g., **Beginner**, **Intermediate**, **Advanced**)
            if current_level is not None:
                formatted_responses.append("\n")  # Add newline before a new level starts
            current_level = stripped_line.replace("**", "")  # Remove bold formatting
            formatted_responses.append(f"{current_level}\n")
        elif stripped_line == "---":  # Detects the end of a question block
            if question_block:  # If a question block exists, send it to the model
                answer_letter = get_response(question_block)
                formatted_responses.append(f"{answer_letter}\n")  # Append answer
                question_block = ""  # Reset for the next question block
        else:
            question_block += line  # Collect everything as part of the current question block

    # Process the last question block if not followed by `---`
    if question_block:
        answer_letter = get_response(question_block)
        formatted_responses.append(f"{answer_letter}\n")

    # Write the formatted responses to the output file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.writelines(formatted_responses)

    print(f"Responses have been written to {output_file}")

# Example usage
input_files = ['./old_doc/md/benchmark_code.md']
temperatures = [0.1]
model_url = 'http://localhost:1234/v1/chat/completions'  # Ensure this is the correct endpoint
model_name = 'lmstudio-community_Meta-Llama-3-8B-Instruct-GGUF'
api_key = 'lm-studio'  # Replace with your actual API key

for questions_file in input_files:
    for temp in temperatures:
        output_file = f"question_and_responses3_{questions_file.split('/')[-1].replace('.md', '')}_{temp}.txt"
        ask(questions_file, output_file, model_url, model_name, api_key, temp)
