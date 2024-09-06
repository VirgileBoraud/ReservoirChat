import requests
import json

def ask(questions_file, output_file, model_url, model_name, api_key, temperature):
    """
    Reads questions from a markdown file, queries an LLM for responses, 
    and saves the responses in a specified markdown file.

    Parameters:
        questions_file (str): Path to the markdown file containing questions.
        output_file (str): Path to the markdown file where responses will be saved.
        model_url (str): The URL of the LLM API endpoint.
        model_name (str): The model name used in the LLM API.
        api_key (str): The API key for accessing the LLM.
        temperature (float): The temperature to use for the LLM responses.
    """

    # Function to get a response from the LLM
    def get_response(question):
        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'}
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": question}],
            "temperature": temperature
        }

        try:
            response = requests.post(model_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()  # Raises an HTTPError if the response code was unsuccessful

            # Try to parse the expected content
            data = response.json()

            # Check if 'choices' and 'message' keys exist in the response
            if 'choices' in data and data['choices']:
                return data['choices'][0]['message'].get('content', 'No content found in the response.')
            else:
                return "Unexpected response structure: " + json.dumps(data)

        except requests.exceptions.RequestException as e:
            print(f"HTTP Request failed: {e}")
            return "Error: Could not retrieve the response due to a request failure."

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
            response_texts = [get_response(question) for _ in range(3)]  # Get three responses
            
            # Append the question and responses side by side in markdown format under the current level
            responses.append(f"## Question: {question}\n### Responses:\n")
            for i, response_text in enumerate(response_texts, 1):
                responses.append(f"**Response {i}:**\n{response_text}\n\n")

    # Write the responses to the output markdown file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.writelines(responses)

    print(f"Responses have been written to {output_file}")

# Example usage
input_files = ['./old_doc/md/questions_specified.md']
temperatures = [0.0,0.2,0.5,0.7]
model_url = 'http://localhost:1234/v1/chat/completions'  # Ensure this is the correct endpoint
model_name = 'lmstudio-community_Meta-Llama-3-8B-Instruct-GGUF'
api_key = 'lm-studio'  # Replace with your actual API key

for questions_file in input_files:
    for temp in temperatures:
        output_file = f"question_and_responses_{questions_file.split('/')[-1].replace('.md', '')}_{temp}.md"
        ask(questions_file, output_file, model_url, model_name, api_key, temp)
