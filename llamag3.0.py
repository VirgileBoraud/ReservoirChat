import requests
import pandas as pd
import numpy as np
from openai import OpenAI
import re
import os
from concurrent.futures import ThreadPoolExecutor
import time
from taipy.gui import Gui, State, notify
import requests
import json

class LLaMag:
    def __init__(self, model_url="http://localhost:1234/v1", embedding_url="http://localhost:1234/v1", api_key="lm-studio", embedding_model="nomic-ai/nomic-embed-text-v1.5-GGUF", model="LM Studio Community/Meta-Llama-3-8B-Instruct-GGUF", message="You are Llamag, a helpful, smart, kind, and efficient AI assistant. You are specialized in reservoir computing. When asked to code, you will code using the reservoirPy library. You will also serve as an interface to a RAG (Retrieval-Augmented Generation) and will use the following documents to respond to your task. DOCUMENTS:", similarity_threshold=0.75, top_n=5):
        self.model_client = OpenAI(base_url=model_url, api_key=api_key)
        self.embedding_client = OpenAI(base_url=embedding_url, api_key=api_key)
        self.embedding_url = embedding_url
        self.embedding_model = embedding_model
        self.model = model
        self.similarity_threshold = similarity_threshold
        self.top_n = top_n
        self.df = self.load_embeddings_from_csv('embeddings.csv')
        self.message = message
        self.code_content = self.read_file('doc/md/codes.md')
        self.history = []
        

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
        
    def get_embeddings(self, text):
        try:
            text = text.replace("\n", " ")
            response = self.embedding_client.embeddings.create(input=[text], model=self.embedding_model)
            print(len(response.data[0].embedding))
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            return None
    
    def get_embedding(self, text):
        try:
            url = self.embedding_url
            headers = {'Content-Type': 'application/json'}
            payload = {'texts': [text.replace("\n", " ")]}
            
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            
            if response.status_code == 200:
                return response.json()[0]
            else:
                print(f"Error getting embeddings: {response.json()}")
                return None
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            return None


    def file_list(self, directory_path):
        try:
            # Get the list of files in the directory
            files = os.listdir(directory_path)
            
            # Filter out directories, only keep files and add the directory path to each file name
            file_names = [os.path.join(directory_path, file) for file in files if os.path.isfile(os.path.join(directory_path, file))]
            
            return file_names
        except Exception as e:
            print(f"An error occurred: {e}")
            return []
    
    def load_data(self, filepaths):
        try:
            with ThreadPoolExecutor() as executor:
                results = executor.map(self.process_file, filepaths)

            all_qa_pairs = [qa for result in results for qa in result]

            if not all_qa_pairs:
                print("No data loaded.")
                return

            self.df = pd.DataFrame(all_qa_pairs)
            self.df['ticket_embedding'] = self.df['ticket'].apply(lambda x: self.get_embedding(x))
            self.df.to_csv('embeddings.csv', index=False)
            print(f"Data saved to embeddings.csv")
        except Exception as e:
            print(f"Error loading data: {e}")

    def process_file(self, filepath):
        all_entries = []
        document_name = os.path.basename(filepath)
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                data = file.read()

            if "Q&A" in filepath:
                # Process as Q&A document
                questions_answers = data.split("Question: ")
                for qa in questions_answers[1:]:
                    parts = qa.split("Answer: ")
                    question = parts[0].strip()
                    answer = parts[1].strip() if len(parts) > 1 else ""
                    all_entries.append({"document": document_name,"ticket": question, "response": answer})

            elif "codes" in filepath:
                # Process as code document with titles
                sections = data.split("## ")
                for section in sections[1:]:
                    lines = section.split('\n', 1)  # Split only on the first newline
                    title = lines[0].strip()
                    if len(lines) > 1:
                        code_block_match = re.search(r'```python(.*?)```', lines[1].strip(), re.DOTALL)
                        if code_block_match:
                            code = f"```python{code_block_match.group(1).strip()}```"
                            all_entries.append({"document": document_name,"ticket": title, "response": code})

            else:
                # Process as regular Markdown document without separating code
                sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', data) ## I have to change things here
                for sentence in sentences:
                    if sentence.strip():
                        all_entries.append({"document": document_name,"ticket": sentence.strip(), "response": sentence.strip()}) ## Use knowledge graph ot create big fan fiction of the docmet 9different concept, family, character, gemerate code to better respond, use this auestion to respond, split document, etc...)

        except Exception as e:
            print(f"Error processing file {filepath}: {e}")
        return all_entries

    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get_top_n_closest_texts(self, text, n=None):
        if n is None:
            n = self.top_n
        query_embedding = self.get_embedding(text)
        self.df.dropna(subset=['ticket_embedding'], inplace=True)
        self.df['similarity'] = self.df['ticket_embedding'].apply(lambda x: self.cosine_similarity(query_embedding, x))
        top_n_indices = self.df[self.df['similarity'] >= self.similarity_threshold].nlargest(n, 'similarity').index
        print(self.df.loc[top_n_indices])
        return self.df.loc[top_n_indices]

    def get_response(self, user_message):
        top_n_results = self.get_top_n_closest_texts(user_message)
        top_n_texts = ("document name: " + top_n_results['document'] + " | ticket: " + top_n_results['ticket'] + " | response: " + top_n_results['response']).tolist()
        # print(top_n_texts)
        system_message = self.message
        system_message += f"History of the previous message: {self.history}"
        '''if "code" in user_message.lower():
            system_message += f"\n\nAdditional Code Content:\n{self.code_content}\n"'''
        system_message += "\n" + str(top_n_texts)
        # print(system_message)
        history = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        completion = self.model_client.chat.completions.create(
            model=self.model,
            messages=history,
            temperature=0.7,
            stream=True,
        )

        response_text = ""
        for chunk in completion:
            if chunk.choices[0].delta.content:
                response_text += chunk.choices[0].delta.content

        self.history.append({"User":user_message,"Document_used":top_n_texts,"LLaMag":response_text})
        return response_text

# Initialize LLaMag instance
system_message = '''You are Llamag, a helpful, smart, kind, and efficient AI assistant. 
        You are specialized in reservoir computing.
        You will serve as an interface to a RAG (Retrieval-Augmented Generation).
        When given documents, you will respond using only these documents. They will serve as inspiration to your reponse.
        If you are not given any document, you will respond that you don't have the response in the database.
        When asked to code, you will use the given additional code content,
        these documents are the only inspiration you can have on the reservoirPy library.
        You will not return the exact responses your were provided, but they will serve as inspiration for your real response.
        You will never use the database you have acquired elsewhere other than in the documents given.
        You will however access the history of the conversation.
        You will use the following documents to respond to your task (You will never display or talk about their existence, only use their information):
        DOCUMENTS:
        (document name : where the name of the document with the similarity are, you will display the name of these documents when you are asked to give the source | ticket: Where the similar questions or embeddings are | answer: the answer or resulting embeddings that will serve you as inspiration to your answer)
        '''

new_message = '''
    You are LLaMag, a helpful, smart, kind and efficient LLM using a RAG (Retrieval-Augmented Generation) application to fetch documents and give relevant answers.
    You are specialised in reservoir computing and in the reservoirPy library.
    You will receive a list of this form:
    (document name :  | ticket:  | answer: )
    document name : is where the name of the document with the similarity are, you will display the name of these documents when you are asked to give the source.
    ticket : are where the similar questions or embeddings are.
    answer : are the answer or resulting embeddings that will serve you as inspiration to your answer.
    If asked to code, you will receive an additional document with a lot of code using the reservoirPy library, you will then code using this codes as references.
    If the documents is empty then you will respond that you don't know.
    DOCUMENTS:
    '''

# llamag = LLaMag(message="system_message")
llamag = LLaMag(model_url='http://localhost:8000/v1', embedding_url='http://localhost:5000/embed', api_key='EMPTY', embedding_model='nomic-ai/nomic-embed-text-v1.5', model='meta-llama/Meta-Llama-3-8B-Instruct', message=system_message, similarity_threshold=0.6, top_n=5)

# Initialize conversation and history
conversation = {
    "Conversation": ["Present Yourself", "Hi! I am LLaMag. The Grrrreat RAG Fetcher, the reservoirPy know-how, the best rider of the High LLaMa tribe, How can I help you today?"]
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
    notify(state, "info", "Sending message...")
    question = state.current_user_message
    state.current_user_message = ""
    history.append({"role": "user", "content": question})
    conv = state.conversation._dict.copy()
    conv["Conversation"].append(question)
    state.conversation = conv
    
    answer = query(state, question).replace("\n", "")
    conv["Conversation"].append(answer)
    state.conversation = conv
    notify(state, "success", "Response received!")

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
# **LL**{: .color-primary}a**M**{: .color-primary}ag # {: .logo-text}
|>

<|part|class_name=p2 align-item-bottom table|
<|{conversation}|table|style=style_conv|show_all|>
<|{current_user_message}|input|label=Write your message here...|on_action=send_message|class_name=fullwidth|change_delay=-1|>
|>
|>

<|part|class_name=markdown|
{markdown_content}
|>
"""

# Run the Taipy GUI
if __name__ == "__main__":
    # llamag.load_data(llamag.file_list('doc/md'))
    Gui(page).run(debug=True, dark_mode=True, use_reloader=True, title="LLaMag")
