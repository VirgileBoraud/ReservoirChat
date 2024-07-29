import subprocess

def execute_graphrag_query():
    # Define the command and arguments
    command = ['python3', '-m', 'graphrag.query', '--root', './ragtest', '--method', 'local', 'what is reservoirPy ?']
    
    # Execute the command
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # Return the output
        return result.stdout
    except subprocess.CalledProcessError as e:
        # If there's an error, return the error message
        return e.stderr

# Example usage:
output = execute_graphrag_query()
print(output)
