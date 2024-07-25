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
import param

class ReservoirChat:
    def __init__(self, model_url="http://localhost:1234/v1", embedding_url="http://localhost:1234/v1", api_key="lm-studio", embedding_model="nomic-ai/nomic-embed-text-v1.5-GGUF", model="LM Studio Community/Meta-Llama-3-8B-Instruct-GGUF", message="You are ReservoirChat, a helpful, smart, kind, and efficient AI assistant. You are specialized in reservoir computing. When asked to code, you will code using the reservoirPy library. You will also serve as an interface to a RAG (Retrieval-Augmented Generation) and will use the following documents to respond to your task. DOCUMENTS:", similarity_threshold=0.75, top_n=5):
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
        self.history = self.load_from_cookies('history', [])
        self.history_history = self.load_from_cookies('history_history', [])
        self.count = 0
        self.day = 0

    def load_from_cookies(self, name, default):
        cookie_value = pn.state.cookies.get(name)
        if cookie_value:
            return json.loads(cookie_value)
        return default

    def save_to_cookies(self, name, value):
        pn.state.cookies[name] = json.dumps(value)

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

        self.history.append({"User":user_message,"Document_used":top_n_texts,"ReservoirChat":response_text})
    
    def cookie():
        from flask import Flask, request, jsonify, make_response

        app = Flask(__name__)

        @app.route('/login', methods=['POST'])
        def login():
            data = request.get_json()
            username = data.get('username')
            password = data.get('password')

            if username == 'testuser' and password == 'testpass':
                res = make_response(jsonify(success=True))
                res.set_cookie('session', 'logged_in')
                return res
            else:
                return jsonify(success=False), 401

        @app.route('/protected')
        def protected():
            session_cookie = request.cookies.get('session')
            if session_cookie == 'logged_in':
                return jsonify(access=True)
            else:
                return jsonify(access=False), 403

        app.run(debug=True)


    def interface(self):
        
        def New_Chat_Interface():
            return pn.chat.ChatInterface(
                            user="Virgile",
                            callback=callback,
                            callback_user="ReservoirChat",
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
                        # pn.pane.HTML(html_content, align='center'),
                        # theme_toggle,
                        chat_interface,
                        # button_properties={"clear": {"callback": clear_history}},
                        sizing_mode='stretch_width'
                        )
        
        def introduction(chat):
            chat.send("When using ReservoirPy Chat, we ask you not to include any personal data about yourself in your exchanges with the system in order to protect your data. The data contained in the chat will be analyzed by Inria",
                    user="Legal Notice",
                    avatar='gavel.png',
                    respond=False
                    )
            chat.send("Hi and welcome! My name is ReservoirChat. I'm a RAG (Retrieval-Augmented Generation) interface specialized in reservoir computing using a Large Language Model (LLM) to help respond to your questions. I will based my responses on a large database consisting of several documents ([See the documents here](https://github.com/VirgileBoraud/ReservoirChat/tree/main/doc/md))",
                    user="ReservoirChat",
                    avatar="logo.png",
                    respond=False
                    )
        
        def new_conversation(event):
            for obj in left.objects:
                if not isinstance(obj, pn.widgets.Button) and self.history == []:
                    return
            
            self.history_history.append(self.history)
            self.save_to_cookies('history_history', self.history_history)
            print(self.history_history)
            new_chat = New_Chat_Interface()
            new_right = right_column(new_chat)

            current_time = time.ctime()
            time_components = current_time.split()
            date_part = " ".join(time_components[0:3])
            time_part = time_components[3]

            layout.pop(1)
            layout.append(new_right)
            introduction(new_chat)

            self.history = []
            for obj in left.objects:
                if isinstance(obj, pn.widgets.Button) and obj.button_type == 'success':
                    obj.button_type = 'primary'
                    return
            
            history_conversation_button = pn.widgets.Button(name=f"{time_part}", button_type='primary', align='center', width=220, height=60) # The name could be the n first letters of the first question of the user
            history_conversation_button.history = self.history_history[-1]  # Store the history in the button

            history_conversation_button.on_click(this_conversation)  # Bind the function
            if self.day == 0:  # Need to change this method to take into account the day
                left.append(pn.pane.Markdown(f"<h2 style='color:white; font-size:18px;'>{date_part}</h2>", align='center'))
                self.day += 1
            left.append(history_conversation_button)

        def this_conversation(event):
            check = 0
            for obj in left.objects:
                if isinstance(obj, pn.widgets.Button) and obj.button_type == 'success':
                    obj.button_type = 'primary'
                    check += 1

            if check == 0:
                if self.history != []:
                    current_time = time.ctime()
                    time_components = current_time.split()
                    date_part = " ".join(time_components[0:3])
                    time_part = time_components[3]
                    history_conversation_button = pn.widgets.Button(name=f"{time_part}", button_type='primary', align='center', width=220, height=60)
                    history_conversation_button.history = self.history  # Store the history in the button
                    history_conversation_button.on_click(this_conversation)
                    if self.day == 0:  # Need to change this method to take into account the day
                        left.append(pn.pane.Markdown(f"<h2 style='color:white; font-size:18px;'>{date_part}</h2>", align='center'))
                        self.day += 1
                    left.append(history_conversation_button)

            event.obj.button_type = 'success'
            button = event.obj
            old_chat = New_Chat_Interface()
            old_right = right_column(old_chat)
            layout.pop(1)
            layout.append(old_right)

            introduction(old_chat)

            list_of_conversation = button.history  # Access the history from the button
            self.history = button.history
            for entry in list_of_conversation:
                if entry:
                    user = entry.get('User')
                    document_used = entry.get('Document_used')
                    reservoirchat = entry.get('ReservoirChat')
                    old_chat.send(user,
                                user="User",
                                respond=False
                                )
                    old_chat.send(reservoirchat,
                                user="ReservoirChat",
                                respond=False
                                )


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

system_message = '''You are ReservoirChat, a helpful, smart, kind, and efficient AI assistant. 
        You are specialized in reservoir computing.
        You will serve as an interface to a RAG (Retrieval-Augmented Generation).
        When given documents, you will respond using only these documents. They will serve as inspiration to your response.
        If you are not given any document, you will only respond : "I didn't found any relevant information about {what the user asked} on the documents I was provided, please refer to the [issue](https://github.com/VirgileBoraud/ReservoirChat/issues) of the github if it should be".
        You will never invent a source if it was not given.
        When asked to code, you will use the given additional code content,
        these documents are the only inspiration you can have to respond to the query of the user.
        You will not return the exact responses your were provided, but they will serve as inspiration for your real response.
        You will never use the database you have acquired elsewhere other than in the documents given.
        You will be given a list in this format:
        {"User": where the previous question of the user is,"Document_used": where the previous documents used are,"ReservoirChat": the response you used for the previous question, as the user can refer to them, you will take them into account for any question of the users.}
        You will never display the two list your are given (The code and DOCUMENTS).
        You will never display the history as well unless the user ask for it.
        You will use the following documents to respond to your task:
        DOCUMENTS:
        (document name : where the name of the document with the similarity are, you will display the name of these documents when you are asked to give the source | ticket: Where the similar questions or embeddings are | answer: the answer or resulting embeddings that will serve you as inspiration to your answer)
        '''

def create_layout():
    return reservoirchat.interface()

if __name__ == "__main__":
    reservoirchat = ReservoirChat(model_url='http://localhost:8000/v1',
                    embedding_url='http://127.0.0.1:5000/v1/embeddings',
                    api_key='EMPTY',
                    embedding_model='nomic-ai/nomic-embed-text-v1.5',
                    model='TechxGenus/Codestral-22B-v0.1-GPTQ',
                    message=system_message,
                    similarity_threshold=0.6,
                    top_n=5)

    file_list = reservoirchat.file_list('doc/md')
    # reservoirchat.load_data(file_list)

    reservoirchat.interface()