import os
import subprocess

def convert(directory):
    # Iterate over all files in the given directory
    for filename in os.listdir(directory):
        # Check if the file is a Markdown file
        if filename.endswith('.md'):
            # Construct the full file path
            file_path = os.path.join(directory, filename)
            # Construct the output text file path
            output_file_path = os.path.splitext(file_path)[0] + '.txt'
            # Use subprocess to call pandoc with the specified arguments
            subprocess.run(['pandoc', file_path, '-f', 'markdown', '-t', 'plain', '-o', output_file_path])
            print(f"Converted {file_path} to {output_file_path}")

convert('doc/papiers_Mnemosyne')