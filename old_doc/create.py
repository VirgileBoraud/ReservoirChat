import os

def extract_code_blocks(file_path):
    """
    Extracts code blocks from the provided file.
    Assumes code is inside triple backticks ``` or indented with 4 spaces.
    """
    code_blocks = []
    title = ""
    in_code_block = False
    code_block = ""

    with open(file_path, 'r') as f:
        for line in f:
            # Check for title
            if line.strip() and not in_code_block and line[0] != ' ':
                title = line.strip()

            # Detect code blocks delimited by triple backticks
            if line.startswith("```"):
                if in_code_block:
                    # Closing triple backticks means the end of a code block
                    code_blocks.append((title, code_block))
                    code_block = ""
                in_code_block = not in_code_block
                continue

            # Detect indented code blocks (typically indented with 4 spaces)
            if line.startswith("    "):
                in_code_block = True
                code_block += line
            elif in_code_block and not line.startswith("    "):
                # End of indented code block
                code_blocks.append((title, code_block))
                in_code_block = False
                code_block = ""
            elif in_code_block:
                code_block += line

    return code_blocks

def process_folder(folder_path, output_file):
    """
    Processes all text files in the folder and appends the code sections to a markdown file.
    """
    with open(output_file, 'w') as md_file:
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".txt"):
                file_path = os.path.join(folder_path, file_name)
                print(f"Processing file: {file_name}")

                code_blocks = extract_code_blocks(file_path)

                if code_blocks:
                    # Write the file name as a section title in markdown format (write once per file)
                    md_file.write(f"```\n# File: {file_name}\n```python\n")

                    for title, code_block in code_blocks:
                        # Append each code block with the title as a markdown comment
                        md_file.write(f"{code_block}\n")

    print(f"Markdown file created at: {output_file}")

# Set folder path and output markdown file
folder_path = "old_doc/rag_training"  # Update with your folder path
output_file = "codes.md"  # Update with your desired output path

process_folder(folder_path, output_file)
