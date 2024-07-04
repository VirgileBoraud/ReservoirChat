import panel as pn
import pandas as pd
import numpy as np
from openai import OpenAI
import re
import os
from concurrent.futures import ThreadPoolExecutor
import time

class LLaMag:
    def __init__(self, base_url, api_key, model="nomic-ai/nomic-embed-text-v1.5-GGUF", chat_model="LM Studio Community/Meta-Llama-3-8B-Instruct-GGUF",message = "You are Llamag, a helpful, smart, kind, and efficient AI assistant. You are specialized in reservoir computing. When asked to code, you will code using the reservoirPy library. You will also serve as an interface to a RAG (Retrieval-Augmneted Generation) and will use the following documents to respond to your task. DOCUMENTS:", similarity_threshold=0.75, top_n=5):
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
    
    def html(self, file_path):
        with open(file_path, 'r') as file:
            markdown_content = file.read()
        # Use regex to find markdown links and extract the URLs
        links = re.findall(r'\[.*?\]\((.*?)\)', markdown_content)
        return links
    
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
    
    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def load_one(self, filepath):
        try:

            all_qa_pairs = self.process_file(filepath)

            if not all_qa_pairs:
                print("No data loaded.")
                return

            df_new = pd.DataFrame(all_qa_pairs)
            df_new['ticket_embedding'] = df_new['ticket'].apply(lambda x: self.get_embedding(x))
            csv_filename = os.path.basename(filepath).replace('.md', '.csv')
            df_new.to_csv(csv_filename, index=False)
            print(f"Data saved to {csv_filename}")
        except Exception as e:
            print(f"Error loading data: {e}")
    
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
                sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', data)
                for sentence in sentences:
                    if sentence.strip():
                        all_entries.append({"document": document_name,"ticket": sentence.strip(), "response": sentence.strip()}) ## Use of llama3 to define question ?

        except Exception as e:
            print(f"Error processing file {filepath}: {e}")
        return all_entries
    
    def get_top_n_closest_texts(self, text, n=None):
        if n is None:
            n = self.top_n
        query_embedding = self.get_embedding(text)
        self.df.dropna(subset=['ticket_embedding'], inplace=True)
        self.df['similarity'] = self.df['ticket_embedding'].apply(lambda x: self.cosine_similarity(query_embedding, x))
        top_n_indices = self.df[self.df['similarity'] >= self.similarity_threshold].nlargest(n, 'similarity').index
        return self.df.loc[top_n_indices]

    def chat(self):
        # Initialize chat history with a system message
        history = [
            {"role": "system", "content": self.message},
            {"role": "user", "content": "Hello, introduce yourself to someone opening this program for the first time. Tell your name, what model and what library you use"},
        ]
        
        print("Chat with an intelligent assistant. Type 'exit' to quit.")
        while True:
            user_input = input("> ")
            if user_input.lower() == "exit":
                break
            
            if self.df is not None and not self.df.empty:
                # Retrieve top N closest texts based on user input
                top_n_results = self.get_top_n_closest_texts(user_input)
                if not top_n_results.empty:
                    top_n_texts = top_n_results['ticket'].tolist()
                    relevant_docs = "\n".join([f"-> Text {i+1}: {t}" for i, t in enumerate(top_n_texts)])
                    if "code" in user_input.lower():
                        relevant_docs += f"\n\nAdditional Code Content:\n{self.code_content}\n"
                    history.append({"role": "system", "content": f"{self.message}\n{relevant_docs}"})

            # Append user input to the history
            history.append({"role": "user", "content": user_input})

            # Generate a response from the model with streaming
            completion = self.client.chat.completions.create(
                model=self.chat_model,
                messages=history,
                temperature=0.7,
                stream=True,
            )

            llamag_message = {"role": "assistant", "content": ""}
            
            # Stream the response and print it out
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    print(chunk.choices[0].delta.content, end="", flush=True)
                    llamag_message["content"] += chunk.choices[0].delta.content

            history.append(llamag_message)
            
            print()

system_message = '''You are Llamag, a helpful, smart, kind, and efficient AI assistant. 
        You are specialized in reservoir computing.
        You will serve as an interface to a RAG (Retrieval-Augmented Generation).
        When given documents, you will respond using only these documents.
        If your are not given any document, you will respond that you don't have the response in the database.
        When ask to code, you will use the given additional code content,
        this document are the only inspiration you can have on the reservoirPy library, you will never use the data you had previously acquired.
        You will never use the database you have acquire elsewhere than in the documents given.
        You will use the following documents to respond to your task.
        DOCUMENTS:
        '''
llamag = LLaMag(base_url="http://localhost:1234/v1", api_key="lm-studio", message=system_message, similarity_threshold=0.75, top_n=5)
file_list = llamag.file_list('doc/md')
# llamag.load_data(file_list)
llamag.chat()