import subprocess

def run_command_for_questions(input_file_path, output_file_path):
    with open(input_file_path, 'r') as file:
        lines = file.readlines()

    # Filter out the questions from the markdown
    questions = [line.strip().split('- ')[1] for line in lines if line.strip().startswith('- ')]

    # Open the output file in write mode
    with open(output_file_path, 'w') as outfile:
        for question in questions:
            command = f'python3 -m graphrag.query --root ragtest --method local "{question}"'
            print(f"Running command: {command}")
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            
            # Write the question and its response to the output file
            outfile.write(f"## Question: {question}\n")
            outfile.write(f"### Response:\n")
            outfile.write(result.stdout)
            outfile.write("\n\n")

if __name__ == "__main__":
    input_file_path = 'doc/md/questions.md'
    output_file_path = 'question_and_responses_local.md'
    run_command_for_questions(input_file_path, output_file_path)