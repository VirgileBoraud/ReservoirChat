import requests
import pandas as pd
import numpy as np
from openai import OpenAI
import re
import os
from concurrent.futures import ThreadPoolExecutor
import time
from taipy.gui import Gui, State, notify

class LLaMag:
    def __init__(self, base_url, api_key, model="nomic-ai/nomic-embed-text-v1.5-GGUF", chat_model="LM Studio Community/Meta-Llama-3-8B-Instruct-GGUF", message="You are Llamag, a helpful, smart, kind, and efficient AI assistant. You are specialized in reservoir computing. When asked to code, you will code using the reservoirPy library. You will also serve as an interface to a RAG (Retrieval-Augmented Generation) and will use the following documents to respond to your task. DOCUMENTS:", similarity_threshold=0.75, top_n=5):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.chat_model = chat_model
        self.similarity_threshold = similarity_threshold
        self.top_n = top_n
        self.df = self.load_embeddings_from_csv('embeddings.csv')
        self.message = message
        self.code_content = self.read_file('doc/md/codes.md')

    def read_file(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return ""
    
    def load_embeddings_from_csv(self, csv_file):
        try:
            df = pd.read_csv(csv_file)
            df['ticket_embedding'] = df['ticket_embedding'].apply(eval).apply(np.array)
            return df
        except Exception as e:
            print(f"Error loading embeddings from CSV: {e}")
            return pd.DataFrame()
        
    def get_embedding(self, text):
        try:
            text = text.replace("\n", " ")
            response = self.client.embeddings.create(input=[text], model=self.model)
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            return None
    
    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get_top_n_closest_texts(self, text, n=None):
        if n is None:
            n = self.top_n
        query_embedding = self.get_embedding(text)
        self.df.dropna(subset=['ticket_embedding'], inplace=True)
        self.df['similarity'] = self.df['ticket_embedding'].apply(lambda x: self.cosine_similarity(query_embedding, x))
        top_n_indices = self.df[self.df['similarity'] >= self.similarity_threshold].nlargest(n, 'similarity').index
        return self.df.loc[top_n_indices]

    def get_response(self, user_message):
        top_n_results = self.get_top_n_closest_texts(user_message)
        top_n_texts = top_n_results['ticket'].tolist()
        system_message = self.message
        if "code" in user_message.lower():
            system_message += f"\n\nAdditional Code Content:\n{self.code_content}\n"
        system_message += "\n".join([f"-> Text {i+1}: {t}" for i, t in enumerate(top_n_texts)])
        history = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        completion = self.client.chat.completions.create(
            model=self.chat_model,
            messages=history,
            temperature=0.7,
            stream=True,
        )

        response_text = ""
        for chunk in completion:
            if chunk.choices[0].delta.content:
                response_text += chunk.choices[0].delta.content

        return response_text

# Initialize LLaMag instance
system_message = '''You are Llamag, a helpful, smart, kind, and efficient AI assistant. 
        You are specialized in reservoir computing.
        You will serve as an interface to a RAG (Retrieval-Augmented Generation).
        When given documents, you will respond using only these documents.
        If you are not given any document, you will respond that you don't have the response in the database.
        When asked to code, you will use the given additional code content,
        these documents are the only inspiration you can have on the reservoirPy library, you will never use the data you had previously acquired.
        You will never use the database you have acquired elsewhere other than in the documents given.
        You will use the following documents to respond to your task.
        DOCUMENTS:
        '''
llamag = LLaMag(base_url="http://localhost:1234/v1", api_key="lm-studio", message=system_message, similarity_threshold=0.75, top_n=5)

# Initialize conversation and history
conversation = {
    "Conversation": ["Present Yourself", "Hi! I am LLaMag. The Grrrreat RAG Fetcher, the reservoirPy know-how, the best rider of the High LlaMa tribe, How can I help you today?"]
}
current_user_message = ""
history = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": " "},
]

# Query function using LLaMag's get_response method
def query(state: State, prompt: str) -> str:
    return llamag.get_response(prompt)

# Function to send a message and get a response
def send_message(state: State) -> None:
    history.append({"role": "user", "content": state.current_user_message})
    answer = query(state, state.current_user_message).replace("\n", "")
    conv = state.conversation._dict.copy()
    conv["Conversation"] += [state.current_user_message, answer]
    state.conversation = conv
    state.current_user_message = ""

# Function to style the conversation
def style_conv(state: State, idx: int, row: int) -> str:
    if idx is None:
        return None
    elif idx % 2 == 0:
        return "user_message"
    else:
        return "llamag_message"

# Define the GUI page
page = """
<|layout|columns=300px 1|
<|part|class_name=sidebar|
# LLaMag **Chat**{: .color-primary} # {: .logo-text}
|>

<|part|class_name=p2 align-item-bottom table|
<|{conversation}|table|style=style_conv|show_all|>
<|{current_user_message}|input|label=Write your message here...|on_action=send_message|class_name=fullwidth|>
|>
|>
"""

# Run the Taipy GUI
if __name__ == "__main__":
    Gui(page).run(debug=True, dark_mode=True, use_reloader=True, title="LLaMag")
