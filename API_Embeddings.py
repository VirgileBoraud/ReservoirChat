from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModel
import torch

app = Flask(__name__)

# Load the model and tokenizer
MODEL_NAME = 'nomic-ai/nomic-embed-text-v1.5'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)

def get_embedding(text):
    text = text.replace("\n", " ")  # Strip newlines as in the LM Studio method
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()  # Extract the embedding and convert to list
    return embedding

@app.route('/v1/embeddings', methods=['POST'])
def embed_text():
    data = request.json
    text = data.get('input')

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        embedding = get_embedding(text)
        # Mimic LM Studio response structure
        response = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": embedding,
                    "index": 0
                }
            ],
            "model": MODEL_NAME,
            "usage": {
                "prompt_tokens": len(tokenizer.tokenize(text)),
                "total_tokens": len(tokenizer.tokenize(text))
            }
        }
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
