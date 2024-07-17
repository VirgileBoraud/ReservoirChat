import requests
import json

def get_embeddings(texts):
    url = 'http://localhost:5000/embed'
    headers = {'Content-Type': 'application/json'}
    payload = {'texts': texts}
    
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    
    if response.status_code == 200:
        return response.json()[0]
    else:
        return {"error": response.json()}

if __name__ == '__main__':
    texts = ['Nomic Embedding API', '#keepAIOpen']
    embeddings = get_embeddings(texts)
    print(embeddings)