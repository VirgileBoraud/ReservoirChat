import panel as pn
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from openai import OpenAI
import re
import os
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

    def load_data(self, filepaths):
        def process_file(filepath):
            all_qa_pairs = []
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    data = file.read()
                
                if "Q&A" in filepath:
                    questions_answers = data.split("Question: ")
                    for qa in questions_answers[1:]:
                        parts = qa.split("Answer: ")
                        question = parts[0].strip()
                        answer = parts[1].strip() if len(parts) > 1 else ""
                        all_qa_pairs.append({"question": question, "answer": answer})
                else:
                    # Assume entire content as a single answer
                    all_qa_pairs.append({"question": "Document Content", "answer": data.strip()})
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

    def find_top_similar_questions(self, query):
        try:
            query_embedding = self.get_embedding(query)
            self.df['similarity'] = self.df['question_embedding'].apply(lambda x: 1 - cosine(query_embedding, x))
            top_similarities = self.df.nlargest(self.top_n, 'similarity')
            top_similarities['similarity_percentage'] = top_similarities['similarity'] * 100
            return top_similarities
        except Exception as e:
            print(f"Error finding top similar questions: {e}")
            return pd.DataFrame()

    def is_coding_request(self, query):
        coding_keywords = ['code']
        return any(keyword in query.lower() for keyword in coding_keywords)

    def get_llm_answer(self, prompt, context=""):
        try:
            messages = [
                {"role": "system", "content": "You are the dumb version of Dumbledore"},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": context}
            ]

            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                temperature=0.7,
                stream=True
            )
            
            new_message = {"role": "assistant", "content": ""}
            for chunk in response:
                if chunk.choices[0].delta.content:
                    new_message["content"] += chunk.choices[0].delta.content

            return new_message["content"]
        except Exception as e:
            print(f"Error getting LLM answer: {e}")
            return "Error generating answer from LLM."

    def get_answer(self, query):
        top_similar_questions = self.find_top_similar_questions(query)
        context = "\n".join([f"Q: {row['question']}\nA: {row['answer']}" for idx, row in top_similar_questions.iterrows()])
        answer = self.get_llm_answer(query, context)
        return answer, top_similar_questions

    def respond(self, query):
        answer, similar_responses = self.get_answer(query)
        response = f"Query: {query}\nAnswer: {answer}"
        if not similar_responses.empty:
            response += "\n\nTop Similar Responses:"
            for index, sim_response in similar_responses.iterrows():
                response += f"\nSimilarity: {sim_response['similarity_percentage']:.2f}%\nQuestion: {sim_response['question']}\nAnswer: {sim_response['answer']}\n"
        return response

    def interactive_query(self, query):
        loading_gif = pn.pane.GIF('https://media.giphy.com/media/3oEjI6SIIHBdRxXI40/giphy.gif', width=100, height=100)
        response_pane = pn.pane.Markdown("")

        def update_response():
            response = self.respond(query)
            response_pane.object = response

        with ThreadPoolExecutor() as executor:
            future = executor.submit(update_response)
            while not future.done():
                response_pane.object = loading_gif.object

        return response_pane

    def chat(self):
        history = [
            {"role": "system", "content": "You are Llamag, a helpful, smart, kind, and efficient AI assistant. You are specialized in reservoir computing. When asked to code, you will code using the reservoirPy library. You will also serve as an interface to a RAG including pre-made questions and responses, and documentation from the reservoirPy library."},
            {"role": "user", "content": "Hello, introduce yourself to someone opening this program for the first time. Be concise."},
        ]

        print("Chat with an intelligent assistant. Type 'exit' to quit.")
        while True:
            completion = self.client.chat.completions.create(
                model=self.chat_model,
                messages=history,
                temperature=0.7,
                stream=True,
            )

            new_message = {"role": "assistant", "content": ""}
            
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    print(chunk.choices[0].delta.content, end="", flush=True)
                    new_message["content"] += chunk.choices[0].delta.content

            history.append(new_message)
            
            print()
            user_input = input("> ")
            if user_input.lower() == "exit":
                break
            history.append({"role": "user", "content": user_input})

    def interface(self):
        input_query = pn.widgets.TextInput(name='Enter your query:')
        output = pn.bind(self.interactive_query, query=input_query)

        layout = pn.Column(
            pn.pane.Markdown("# LLaMag"),
            input_query,
            output
        )

        pn.serve(layout, start=True)

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

llamag = LLaMag(base_url="http://localhost:1234/v1", api_key="lm-studio", top_n=5)
#print(llamag.html('doc/Q&A_format.md'))
directory_path = 'doc/md'
file_list = llamag.file_list(directory_path)
llamag.load_data(file_list)
llamag.chat()
