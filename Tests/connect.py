import requests

# The URL you want to connect to
url = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"

try:
    # Make a GET request to the URL
    response = requests.get(url, timeout=10)  # Timeout is set to 10 seconds

    # Check if the request was successful
    if response.status_code == 200:
        print("Connection successful.")
        # Do something with the response, e.g., print or save the content
        print(response.content)
    else:
        print(f"Failed to connect. Status code: {response.status_code}")

except requests.exceptions.ConnectTimeout:
    print("Connection timed out.")

except requests.exceptions.RequestException as e:
    # Catch all other request-related exceptions
    print(f"An error occurred: {e}")
