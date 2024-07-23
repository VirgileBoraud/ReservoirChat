import panel as pn
import pandas as pd
import numpy as np
from openai import OpenAI
import re
import os
from concurrent.futures import ThreadPoolExecutor
import requests
import json
from panel.chat import ChatMessage
from http.cookies import SimpleCookie
from datetime import date, datetime
import time

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
        self.history_history = []

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
        
    # def get_embeddings(self, text):
    #     try:
    #         text = text.replace("\n", " ")
    #         response = self.embedding_client.embeddings.create(input=[text], model=self.embedding_model)
    #         print(len(response.data[0].embedding))
    #         return response.data[0].embedding
    #     except Exception as e:
    #         print(f"Error getting embeddings: {e}")
    #         return None
    
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

    def get_response(self, user_message, history):
        top_n_results = self.get_top_n_closest_texts(user_message)
        top_n_texts = ("document name: " + top_n_results['document'] + " | ticket: " + top_n_results['ticket'] + " | response: " + top_n_results['response']).tolist()
        # print(top_n_texts)
        system_message = self.message
        system_message += f"History of the previous message: {history}"
        '''if "code" in user_message.lower():
            system_message += f"\n\nAdditional Code Content:\n{self.code_content}\n"'''
        system_message += "\n" + str(top_n_texts)
        # print(system_message)
        n_history = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        completion = self.model_client.chat.completions.create(
            model=self.model,
            messages=n_history,
            temperature=0.7,
            stream=True,
        )

        response_text = ""
        for chunk in completion:
            if chunk.choices[0].delta.content:
                response_text += chunk.choices[0].delta.content
                yield response_text

        self.history.append({"User":user_message,"Document_used":top_n_texts,"LLaMag":response_text})

    def interface(self):
        
        def New_Chat_Interface():
            return pn.chat.ChatInterface(
                            # ChatMessage("Hi and welcome! My name is LLaMag, I'm a Large Language Model (LLM), serving as a RAG (Retrieval-Augmented Generation) interface, specialised in reservoir computing. I will based my responses on a large database consisting of several documents ([See the documents here](https://github.com/VirgileBoraud/Llamag/tree/main/doc/md))"),
                            user="Virgile",
                            callback=callback,
                            callback_user="LLaMag",
                            # callback_avatar="Sir_Gillington.png",
                            # widgets=pn.widgets.FileInput(name="CSV File", accept=".csv"), To add the possibility to add documents
                            # reset_on_send=False, for ne reset of writing
                        )
        
        def layout(left, right):
            return pn.Row(
                    left,
                    right
                    )
        
        def left_column():
            return pn.Column(
                        pn.pane.Markdown("<h2 style='color:white; font-size:28px;'>Conversations</h2>", align='center'),
                        new_conversation_button,
                        styles={'padding': '10px', 'border': '2px solid #2FA6BD','background': '#31ABC7'},
                        width=300,
                        sizing_mode='stretch_height'
                        )
        
        def right_column(chat_interface):
            return pn.Column(
                        pn.pane.HTML("""
                        <div style="display: flex; align-items: center;">
                            <h2 style='font-size:32px;'>Reservoir</h2><h2 style='color:#31ABC7; font-size:32px;'>Chat</h2>
                        </div>
                        """, align='center'),
                        #pn.pane.HTML(html_content, align='center'),
                        # theme_toggle,
                        chat_interface,
                        # button_properties={"clear": {"callback": clear_history}},
                        sizing_mode='stretch_width'
                        )
        
        def introduction(chat):
            chat.send("Hi and welcome! My name is Sir Gillington. I'm a RAG (Retrieval-Augmented Generation) interface specialised in reservoir computing using a Large Language Model (LLM) to help respond to your questions. I will based my responses on a large database consisting of several documents ([See the documents here](https://github.com/VirgileBoraud/Llamag/tree/main/doc/md))",
                    user="Sir Gillington",
                    avatar="Sir_Gillington.png",
                    respond=False
                    )
        
        def this_conversation(event):
            old_chat = New_Chat_Interface()
            old_right = right_column(old_chat)
            layout.pop(1)
            layout.append(old_right)
            list_history_history = self.history_history
            list_of_conversation = list_history_history[1] # I need to change some things here because i doesn't show the right conversation when clicking
            print(list_of_conversation)
            for i in list_of_conversation:
                old_chat.send(
                            "Hi and welcome! My name is Sir Gillington. I'm a RAG (Retrieval-Augmented Generation) interface specialised in reservoir computing using a Large Language Model (LLM) to help respond to your questions. I will based my responses on a large database consisting of several documents ([See the documents here](https://github.com/VirgileBoraud/Llamag/tree/main/doc/md))",
                            user="Sir Gillington",
                            avatar="Sir_Gillington.png",
                            respond=False
                            )

        def new_conversation(event):
            if not event:
                return
            
            self.history_history.append(self.history)
            print(self.history_history)
            new_chat = New_Chat_Interface()
            new_right = right_column(new_chat)

            current_time = time.ctime()
            time_components = current_time.split()
            date_part = " ".join(time_components[0:4])
            time_part = time_components[3]

            llm_message = [
                            {"role": "system", "content": f"Resume the questions asked by the user in less than 5 words in this conversation : {self.history}"},
                            {"role": "user", "content": "Resume it"},
                           ]
            
            completion = self.model_client.chat.completions.create(
                                                                    model=self.model,
                                                                    messages=llm_message,
                                                                    temperature=0.7,
                                                                    stream=True,
                                                                    )
            resume_llm = ""

            layout.pop(1)
            layout.append(new_right)
            introduction(new_chat)

            for chunk in completion:
                if chunk.choices[0].delta.content:
                    resume_llm += chunk.choices[0].delta.content

            self.history = []
            history_conversation_button = pn.widgets.Button(name=f"{time_part}", button_type='primary', align='center', width=220, height=60, description=f"{resume_llm}")
            pn.bind(this_conversation, history_conversation_button, watch=True)
            left.append(history_conversation_button)

        def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
            # To avoid having too much history
            if len(self.history) >= 10:
                self.history.pop(3)
                print(self.history)
            return self.get_response(contents, self.history)

        def clear_history(instance, event):
            self.history.clear()

        new_conversation_button = pn.widgets.Button(name='New Conversation', button_type='primary', align='center')
        pn.bind(new_conversation, new_conversation_button, watch=True)
        
        chat_interface = New_Chat_Interface()
        left = left_column()
        right = right_column(chat_interface)

        introduction(chat_interface)
        
        layout = layout(left, right)
        
        pn.serve(layout, title="ReservoirChat", port=8080)

system_message = '''You are Llamag, a helpful, smart, kind, and efficient AI assistant. 
        You are specialized in reservoir computing.
        You will serve as an interface to a RAG (Retrieval-Augmented Generation).
        When given documents, you will respond using only these documents. They will serve as inspiration to your reponse.
        If you are not given any document, you will respond that you don't have the response in the database.
        When asked to code, you will use the given additional code content,
        these documents are the only inspiration you can have on the reservoirPy library.
        You will not return the exact responses your were provided, but they will serve as inspiration for your real response.
        You will never use the database you have acquired elsewhere other than in the documents given.
        You will be given a list in this format:
        {"User": where the previous question of the user is,"Document_used": where the previous documents used are,"LLaMag": the response you used for the previous question, as the user can refer to them, you will take them into account for any question of the users.}
        You will never display the two list given
        You will use the following documents to respond to your task:
        DOCUMENTS:
        (document name : where the name of the document with the similarity are, you will display the name of these documents when you are asked to give the source | ticket: Where the similar questions or embeddings are | answer: the answer or resulting embeddings that will serve you as inspiration to your answer)
        '''

def create_layout():
    return llamag.interface()

if __name__ == "__main__":
    llamag = LLaMag(model_url='http://localhost:8000/v1',
                    embedding_url='http://127.0.0.1:5000/v1/embeddings',
                    api_key='EMPTY',
                    embedding_model='nomic-ai/nomic-embed-text-v1.5',
                    model='TechxGenus/Codestral-22B-v0.1-GPTQ',
                    message=system_message,
                    similarity_threshold=0.6,
                    top_n=5)

    file_list = llamag.file_list('doc/md')
    # llamag.load_data(file_list)

    llamag.interface()