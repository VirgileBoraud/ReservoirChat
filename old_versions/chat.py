import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os

class LLaMag:
    def __init__(self, api_key, base_url="http://localhost:1234/v1", embedding_model="nomic-ai/nomic-embed-text-v1.5-GGUF"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.embedding_model = embedding_model
        self.history = [
            {"role": "system", "content": "You are Llamag, a helpful, smart, kind, and efficient AI assistant. You are specialized in reservoir computing. When asked to code, you will code using the reservoirPy library. You will also serve as an interface to a RAG including pre-made questions and responses, and documentation from the reservoirPy library."},
            {"role": "user", "content": "Hello, introduce yourself to someone opening this program for the first time. Be concise."},
        ]
        self.document_embeddings = []
        self.documents = []

    def get_embedding(self, text):
        text = text.replace("\n", " ")
        response = self.client.embeddings.create(input=[text], model=self.embedding_model)
        return response.data[0].embedding

    def add_documents(self, documents):
        self.documents = documents
        self.document_embeddings = [self.get_embedding(doc) for doc in documents]

    def find_most_similar_documents(self, query_embedding, top_n=3):
        similarities = cosine_similarity([query_embedding], self.document_embeddings)[0]
        most_similar_indices = similarities.argsort()[-top_n:][::-1]
        return most_similar_indices, similarities[most_similar_indices]

    def load_documents_from_files(self, filepaths):
        documents = []
        for filepath in filepaths:
            with open(filepath, 'r', encoding='utf-8') as file:
                documents.append(file.read())
        self.add_documents(documents)

    def chat(self):
        print("Starting interactive chat. Type your message and press Enter. Type 'exit' to end the chat.")
        while True:
            user_input = input("> ")
            if user_input.lower() == "exit":
                break
            
            self.history.append({"role": "user", "content": user_input})
            
            # Get embedding for the user question
            query_embedding = self.get_embedding(user_input)
            
            # Find most similar documents
            most_similar_indices, _ = self.find_most_similar_documents(query_embedding, top_n=5)
            relevant_docs = [self.documents[idx] for idx in most_similar_indices]

            # Generate a message based on the most relevant documents
            context_message = " ".join(relevant_docs)
            self.history.append({"role": "system", "content": "Consider the following context: " + context_message})
            
            completion = self.client.chat.completions.create(
                model="LM Studio Community/Meta-Llama-3-8B-Instruct-GGUF",
                messages=self.history,
                temperature=0.7,
                stream=True,
            )

            new_message = {"role": "assistant", "content": ""}
            
            for chunk in completion:
                if chunk.choices[0].delta and chunk.choices[0].delta.content:
                    print(chunk.choices[0].delta.content, end="", flush=True)
                    new_message["content"] += chunk.choices[0].delta.content

            self.history.append(new_message)
            print()

    def file_list(self, directory_path):
        try:
            files = os.listdir(directory_path)
            file_names = [os.path.join(directory_path, file) for file in files if os.path.isfile(os.path.join(directory_path, file))]
            return file_names
        except Exception as e:
            print(f"An error occurred: {e}")
            return []

# Example usage
api_key = "lm-studio"
llamag = LLaMag(api_key='lm-studio')
file_list = llamag.file_list('old_doc/md')
# llamag.load_documents_from_files(file_list)
llamag.chat()