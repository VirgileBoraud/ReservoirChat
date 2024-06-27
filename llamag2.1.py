import panel as pn
import os
import pandas as pd
import numpy as np
from openai import OpenAI
import re
from concurrent.futures import ThreadPoolExecutor

class LLaMag:
    def __init__(self, base_url, api_key, model="nomic-ai/nomic-embed-text-v1.5-GGUF", chat_model="LM Studio Community/Meta-Llama-3-8B-Instruct-GGUF", similarity_threshold=75, top_n=5):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.chat_model = chat_model
        self.similarity_threshold = similarity_threshold
        self.top_n = top_n
        self.df = None

    def get_embedding(self, text):
        try:
            text = text.replace("\n", " ")
            response = self.client.embeddings.create(input=[text], model=self.model)
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None

    def load(self, filepaths):
        def process_file(filepath):
            all_qa_pairs = []
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
                        combined_text = f"Question: {question} Answer: {answer}"
                        all_qa_pairs.append({"question": combined_text, "answer": answer})
                else:
                    # Process as regular Markdown document
                    code_blocks = re.findall(r'```(.*?)```', data, re.DOTALL)
                    data_no_code = re.sub(r'```.*?```', '', data, flags=re.DOTALL)
                    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', data_no_code)

                    for block in code_blocks:
                        all_qa_pairs.append({"question": block.strip(), "answer": block.strip()})

                    for sentence in sentences:
                        if sentence.strip():
                            all_qa_pairs.append({"question": sentence.strip(), "answer": sentence.strip()})
            except Exception as e:
                print(f"Error processing file {filepath}: {e}")
            return all_qa_pairs

        try:
            with ThreadPoolExecutor() as executor:
                results = executor.map(process_file, filepaths)

            all_qa_pairs = [qa for result in results for qa in result]

            if not all_qa_pairs:
                print("No data loaded.")
                return

            self.df = pd.DataFrame(all_qa_pairs)
            self.df['question_embedding'] = self.df['question'].apply(lambda x: self.get_embedding(x))
            self.df.to_csv('qa_embeddings.csv', index=False)
        except Exception as e:
            print(f"Error loading data: {e}")

    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get_top_n_closest_texts(self, text, n=None):
        if n is None:
            n = self.top_n
        query_embedding = self.get_embedding(text)
        similarities = [self.cosine_similarity(query_embedding, embedding) for embedding in self.df['question_embedding']]
        top_n_indices = np.argsort(similarities)[::-1][:n]
        return self.df.iloc[top_n_indices]

    def get_response(self, user_message):
        if self.df is None or self.df.empty:
            return "No data available to start chat. Please load data first."

        top_n_results = self.get_top_n_closest_texts(user_message)
        if top_n_results.empty:
            return "No similar texts found."

        top_n_texts = top_n_results['question'].tolist()

        system_message = "You are Llamag, a helpful, smart, kind, and efficient AI assistant. You are specialized in reservoir computing. Here are the most relevant documents for your query:\n"
        system_message += "\n".join([f"-> Text {i+1}: {t}" for i, t in enumerate(top_n_texts)])

        history = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        # Initialize the response text
        response_text = "LLaMag:\n"
        
        # Use streaming
        for response in self.client.chat.completions.create(
            model=self.chat_model,
            messages=history,
            temperature=0.2,
            stream=True,
        ):
            if response.choices[0].delta.content is not None:
                response_text += response.choices[0].delta.content
        
        return response_text


    def html(self, file_path):
        with open(file_path, 'r') as file:
            markdown_content = file.read()
        links = re.findall(r'\[.*?\]\((.*?)\)', markdown_content)
        return links

    def file_list(self, directory_path):
        try:
            files = os.listdir(directory_path)
            file_names = [os.path.join(directory_path, file) for file in files if os.path.isfile(os.path.join(directory_path, file))]
            return file_names
        except Exception as e:
            print(f"An error occurred: {e}")
            return []

def callback(input_message, input_user, instance: pn.chat.ChatInterface):
    message = llamag.get_response(input_message)
    return message

llamag = LLaMag(base_url="http://localhost:1234/v1", api_key="lm-studio", top_n=5)
file_list = llamag.file_list('doc/md')
llamag.load(file_list)

chat_interface = pn.chat.ChatInterface(callback=callback)
chat_interface.servable()

layout = pn.Column(pn.pane.Markdown(f"## LLaMag", align='center'),chat_interface)

def __panel__(self):
    return layout

if __name__ == "__main__":
    pn.serve(layout)