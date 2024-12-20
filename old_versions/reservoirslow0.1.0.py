import panel as pn
import pandas as pd
import numpy as np
from openai import OpenAI
import re
import os
from concurrent.futures import ThreadPoolExecutor
import requests
import json
import time
import ast
# import subprocess
from graphrag.query.cli import run_global_search
from graphrag.query.cli import run_local_search
import uuid
from pymongo import MongoClient
from datetime import datetime
# Initialize MongoClient
client = MongoClient('localhost', 27017)
db = client['ReservoirChat']
ma = client['MangoChat']
def app():
    class ReservoirChat:

        # Initializing the class
        def __init__(self, model_url="http://localhost:1234/v1", embedding_url="http://localhost:1234/v1", api_key="lm-studio", embedding_model="nomic-ai/nomic-embed-text-v1.5-GGUF", model="LM Studio Community/Meta-Llama-3-8B-Instruct-GGUF", message="You are ReservoirChat, a helpful, smart, kind, and efficient AI assistant. You are specialized in reservoir computing. When asked to code, you will code using the reservoirPy library. You will also serve as an interface to a RAG (Retrieval-Augmented Generation) and will use the following documents to respond to your task. DOCUMENTS:", similarity_threshold=0.75, top_n=5):
            
            # Basic variables
            self.model_client = OpenAI(base_url=model_url, api_key=api_key)
            self.embedding_client = OpenAI(base_url=embedding_url, api_key=api_key)
            self.embedding_url = embedding_url
            self.embedding_model = embedding_model
            self.model = model
            self.similarity_threshold = similarity_threshold
            self.top_n = top_n
            self.message = message

            # The vector store is called, it serve to keep the loaded vector created by the self.load_data function and to not have to load each document each time
            self.df = self.load_embeddings_from_csv('embeddings.csv')
            
            # Read the content of the code documents, to be used by the prompt when we think the user ask to code
            # Serve for better coding output
            self.code_content = self.read_file('doc/md/codes.md')

            # We check if the use of cookies was accepted by the user, if not, self.cookie_check is False
            print(pn.state.cookies.get("check")) # Checking the cookie
            if pn.state.cookies.get("check"):
                self.cookie_check = True
            else:
                self.cookie_check = False
            
            # Load the history the user didn't "saved" into self.history_history because he didn't click New Conversation
            # Not yet implemented
            self.history = self.load_from_cookies('history', [])
            
            if pn.state.cookies.get("uuid"):
                self.history_history = [[]]
                for conversation in collection.find({"user_id": uuid_v4}):
                    self.history_history[0].append(conversation)
                print(self.history_history)

            else:
                self.history_history = self.load_from_cookies('history_history', [])

            # The current time for date and everything related
            current_time = time.ctime()
            time_components = current_time.split()
            self.day = " ".join(time_components[0:3])
            self.check_day = 0 # Little check to be removed when the date method is updated (Currently using only the current day, not the day the conversation was created)
            self.count_pop = 0

        # --------------------------------------------------------------------------------------------------------------------------------------------------
        
        # Used to load cookie, to be used in a cookie approach instead of cache approach in history of conversation
        def load_from_cookies(self, name, default):
            cookie_value = pn.state.cookies.get(name)
            if cookie_value:
                return json.loads(cookie_value)
            return default

        # Used to save cookies, this method as to be combined with layout.append().
        # Example : cookie = self.save_to_cookies("check", True) => layout.append(cookie)
        def save_to_cookies(self, name, value):
            cookie = pn.pane.HTML(f"""
                            <script>
                            document.cookie="{name}={value};expires=" +
                            new Date(Date.now()+(30*24*60*60*1000)).toUTCString() + ";SameSite=Strict";
                            </script>
                            """)
            print(f"Cookie {name} set with value {value}")
            return cookie

        # To read markdown file (used to read the codes.md document)
        def read_file(self, filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    return file.read()
            except Exception as e:
                print(f"Error reading file: {e}")
                return ""
        
        # used to load the vector store stored on a csv_file
        def load_embeddings_from_csv(self, csv_file):
            try:
                df = pd.read_csv(csv_file)
                df['ticket_embedding'] = df['ticket_embedding'].apply(eval).apply(np.array)
                return df
            except Exception as e:
                print(f"Error loading embeddings from CSV: {e}")
                return pd.DataFrame()
        
        # Used to get embeddings from a documents
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

        # Get the list of files in a specified directory path (used to load the data of a list of documents)
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

        # Process a file (different files have different process) to create a list of JSON objects
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
        
        # Load data from a list of documents
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
                print("Data saved to embeddings.csv")
            except Exception as e:
                print(f"Error loading data: {e}")
    
        # The cosine similarity function used when comparing 2 vectors
        def cosine_similarity(self, a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        # Returns the top n closest texts. Is used to compare the query of a user to embeddings of text.
        def get_top_n_closest_texts(self, text, n=None):
            if n is None:
                n = self.top_n
            query_embedding = self.get_embedding(text)
            self.df.dropna(subset=['ticket_embedding'], inplace=True)
            self.df['similarity'] = self.df['ticket_embedding'].apply(lambda x: self.cosine_similarity(query_embedding, x))
            top_n_indices = self.df[self.df['similarity'] >= self.similarity_threshold].nlargest(n, 'similarity').index
            print(self.df.loc[top_n_indices])
            return self.df.loc[top_n_indices]

        # Function to get the response (with history of current conversation) from the query of the user
        def get_responses(self, user_message, history):
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
                temperature=0.1,
                stream=True,
            )

            response_text = ""
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    response_text += chunk.choices[0].delta.content
                    # For streaming purposes (which means real time response), we need to yield the response
                    yield response_text

            self.history.append({"User":user_message,"Document_used":top_n_texts,"ReservoirChat":response_text})

        # Function to insert or update conversations from self.history_history
        def mongo(self, history_history):
            current_time = time.ctime()
            time_components = current_time.split()
            custom_date = " ".join(time_components[0:3])
            collection.delete_many({})
            for conversation in history_history:
                
                # Add timestamps to each interaction
                interactions_with_timestamps = []
                for interaction in conversation:
                    interaction_with_timestamp = {
                        "User": interaction["User"],
                        "ReservoirChat": interaction["ReservoirChat"],
                        "timestamp": time.time()  # Current time in UTC for precision
                    }
                    interactions_with_timestamps.append(interaction_with_timestamp)
                
                # Prepare the document for insertion with the custom date format
                conversation_doc = {
                    "user_id": uuid_v4,  # User-specific ID
                    "date": custom_date,  # Custom formatted date
                    "interactions": interactions_with_timestamps
                }
            
                # Insert the conversation document into the collection
                collection.insert_one(conversation_doc)
                print('COLLECTION INSERTED')
                for conversation in collection.find({"user_id": uuid_v4}):
                    print(conversation)
        
        def mango(self, history):
            current_time = time.ctime()
            time_components = current_time.split()
            custom_date = " ".join(time_components[0:3])
            document = {
                "UserID": uuid_v4,
                "Date": custom_date,
                "History": history,
            }
            mangollection.insert_one(document)
            print(f"Document {document} inserted")
            documents = mangollection.find()
            for doc in documents:
                print(f"UserID: {doc.get('UserID')}, Date: {doc.get('Date')}, History: {doc.get('History')}")
        
        def get_response(self, user_message, history):
            print('--------------Get_Response History-----------------')
            print(history)
            print('---------------------------------------------------')
            if history != []:
                if not isinstance(history[0], dict):
                    history = history[0]
                    print('--------------Get_Response History Check-----------------')
                    print(history)
                    print('---------------------------------------------------')
            message = run_local_search('ragtest',
                                          'ragtest/output/everything4/artifacts',
                                          'ragtest',
                                          0,
                                          'This is a response',
                                          False,
                                          user_message,
                                          history)
            response_text = ""
            for chunk in message:
                response_text += chunk
                # For streaming purposes (which means real time response), we need to yield the response
                yield response_text
                time.sleep(0.01)
            '''system_message = f
                    You are ReservoirChat, a helpful, smart, kind, and efficient AI assistant.
                    You are specialized in reservoir computing.
                    If a user ask a question in a language, you will respond in this language.
                    You will receive 2 things :
                    1. A history of conversation that will serve as your memory of the conversation
                    2. A text that will serve a documentation. Your general will not be used, only this text

                    You will then remove data of references from the text you received and take into account the history of conversation.

                    If you consider that the user message is only related to the history of conversation, use only the history of conversation

                    HISTORY OF CONVERSATION:
                    {history}

                    DOCUMENTATION:
                    {message}
                    
            prompt = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ]
            completion = self.model_client.chat.completions.create(
                model=self.model,
                messages=prompt,
                temperature=0.1,
                stream=True,
            )

            response_text = ""
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    response_text += chunk.choices[0].delta.content
                    # For streaming purposes (which means real time response), we need to yield the response
                    yield response_text'''
            self.history.append({"User":user_message, "ReservoirChat":response_text})
            if self.cookie_check:
                self.mango({"User":user_message, "ReservoirChat":response_text})
        
        #------------------------------------------------------------------------------------------------------------------------------------------------

        # This is the interface, where everything the user see is coded
        # Note: panel 1.3.7 offer streaming where panel 1.4.4 don't. 1.4.4 offer user personalization which is less important
        def interface(self):
            
            # A function that creates a new Chat Interface where a user can ask a question of demand any information he wants.
            # It is used when we initiate the interface, or when we want a new conversation
            def New_Chat_Interface():
                return pn.chat.ChatInterface(
                                user="User",
                                callback=message_callback,
                                callback_user="ReservoirChat",
                                callback_exception="verbose",
                            )
            
            # This is the callback of a message used if New_Chat_Interface. When a query of the user is sent to the LLM, it returns the response the LLM gave back
            def message_callback(contents: str, user: str, instance: pn.chat.ChatInterface):
                # To avoid having too much history
                if len(self.history) >= 10:
                    self.history.pop(3)
                    print(self.history)
                return self.get_response(contents,self.history)

            # The main layout, where the left part (the history of conversation) if dispose next to the chat_interface (where the user ask questions)
            # and everything else like the title, etc...
            def layout(left, right):
                return pn.Row(
                        left,
                        right
                        )
            
            # The history of conversation column
            def left_column():
                return pn.Column(
                            pn.pane.Markdown("<h2 style='color:white; font-size:28px;'>Conversations</h2>", align='center'),
                            new_conversation_button,
                            styles={'padding': '10px', 'border': '2px solid #2FA6BD','background': '#31ABC7'},
                            width=300,
                            sizing_mode='stretch_height'
                            )
            
            # The Chat interface column
            def right_column(chat_interface):
                return pn.Column(
                    pn.Row(
                        # First column: Empty, can be used for a logo or placeholder
                        pn.Column(
                            pn.Spacer(width=70),  # Adjust width as needed for your layout
                            sizing_mode='stretch_width'
                        ),
                        # Second column: Centered "ReservoirChat" title
                        pn.Column(
                            pn.pane.HTML("""
                            <div style="display: flex; justify-content: center; align-items: center;">
                                <h2 style='font-size:32px;'>Reservoir</h2>
                                <h2 style='color:#31ABC7; font-size:32px;'>Chat</h2>
                            </div>
                            """, align='center', sizing_mode='stretch_width'),
                            sizing_mode='stretch_width'
                        ),
                        # Third column: Images aligned to the right
                        pn.Column(
                            pn.Row(
                                pn.pane.Image("png/reservoirpylogoclean.png", height=60, margin=(20, 20, 20, 20)),
                                pn.pane.Image("png/INRIAlogoclean.png", height=32, margin=(31, 20, 31, 20)),
                                pn.pane.Image("png/ANRlogoclean.png", height=28, margin=(35, 20, 35, 20)),
                                align='end'
                            ),
                            align='end',
                            sizing_mode='stretch_width'
                        ),
                        sizing_mode='stretch_width'
                    ),
                    cookie_popup,
                    chat_interface,
                    sizing_mode='stretch_width'
                )

            # This is the initial information that is displayed. Mainly, the Legal Notice and Main information about ReservoirChat
            def introduction(chat):
                chat.send("ReservoirChat uses up-to-date documentation and code examples from [ReservoirPy](https://github.com/reservoirpy/reservoirpy): a Python library to code reservoir computing based neural networks ([More info](https://github.com/reservoirpy/reservoirpy)). ReservoirChat is a beta tool provided by Mnemosyne lab at Inria, Bordeaux, France. It is supported by Inria ([BrainGPT project](https://www.inria.fr/en/braingpt)) and ANR ([DeepPool project](https://team.inria.fr/mnemosyne/deeppool/)).",
                        user="Inria",
                        avatar='png/INRIAlogoclean.png',
                        respond=False
                        )
                
                # The Legal Notice
                chat.send("When using ReservoirChat, we ask you not to include any personal data in your interactions with the system. The data contained in the chat will be analyzed by Inria.",
                        user="Legal Notice",
                        avatar='png/gavel.png',
                        respond=False
                        )
                # The main information
                chat.send("Hi and welcome! My name is ReservoirChat. I'm a RAG (Retrieval-Augmented Generation) interface specialized in Reservoir Computing using a Large Language Model (LLM) to help respond to your questions. Based on several documents and ReservoirPy documentation, I can answer general questions on Reservoir Computing and generate code. ReservoirChat uses [GraphRag](https://microsoft.github.io/graphrag/) and [Codestral](https://mistral.ai/news/codestral/).",
                        user="ReservoirChat",
                        avatar="png/logo.png",
                        respond=False
                        )

                '''chat.send("Hi and welcome! My name is ReservoirChat. I'm a RAG (Retrieval-Augmented Generation) interface specialized in reservoir computing using a Large Language Model (LLM) to help respond to your questions. I will based my responses on a large database consisting of several documents ([See the documents here](https://github.com/VirgileBoraud/ReservoirChat/tree/main/doc/md))",
                        user="ReservoirChat",
                        avatar="png/logo.png",
                        respond=False
                        )'''
                
                chat.send("ReservoirChat is currently being fine-tuned with additional data, so the service may be unusually slow. Please come back later if it's too slow.",
                        user="Work In Progress",
                        avatar="png/constructionconeclean.png",
                        respond=False
                        )
            
            # The Function that is executed when clicking on the New_Conversation Button
            def new_conversation(event):
                # For each objects (mainly button here) in the left column
                for obj in left.objects:
                    # If no conversation exists and if nothing was written by the user (=Thus no history where created)
                    if not isinstance(obj, pn.widgets.Button) and self.history == []:
                        # We do nothing as it could create a void conversation which serve to nothing
                        return
                
                # Creating a new right column (so a new chat interface)
                new_chat = New_Chat_Interface()
                new_right = right_column(new_chat)

                # We remove the old one and add a new one with an introduction
                layout.pop(1)
                if self.count_pop == 0:
                    self.count_pop += 1
                    layout.pop(1)

                layout.append(new_right)
                introduction(new_chat)

                # For each conversations
                for obj in left.objects:
                    # If a button is detected and it is green
                    if isinstance(obj, pn.widgets.Button) and obj.button_type == 'success':
                        # Make it blue again and reset the history, return to stop the process
                        # If we do not do that, the green button will remain green (Thus highlighting the current conversation is impossible))
                        # And we wan't to change conversation at will, without importing the current history
                        obj.button_type = 'primary'
                        self.history = []
                        return
                

                self.history_history[0].append(self.history)
                print('----------history_history------------')
                print(self.history_history)
                print('-------self.history_history[0]----------')
                print(self.history_history[0])
                print('------------------------------------------------------')
                print('-------self.history_history[0][0]----------')
                print(self.history_history[0][0])
                print('------------------------------------------------------')

                # If cookie was accepted
                if self.cookie_check:
                    # import the history of history into the cache
                    # pn.state.cache['history_history'] = self.history_history
                    # Saving history_history in the Mongo DataBase
                    # self.mongo("ReservoirChat", "history_history", self.history_history)
                    self.mongo(self.history_history[0])
                    for conversation in collection.find({"user_id": uuid_v4}):
                        print(conversation)

                # The name variable if the 6 first words the user wrote, it will be used to name the button
                name = self.history[0]['User']
                name = name.split()
                name = ' '.join(name[:6])

                self.history = []

                # Creating the previous conversation button
                history_conversation_button = pn.widgets.Button(name=f"{name}", button_type='primary', align='center', width=220, height=60)
                history_conversation_button.history = self.history_history[-1] # Store the history in the button

                history_conversation_button.on_click(this_conversation) # Bind the function this_conversation to the click of the history_conversation_button
                
                if self.check_day == 0: # Need to change this method to take into account the day
                    left.append(pn.pane.Markdown(f"<h2 style='color:white; font-size:18px;'>{self.day}</h2>", align='center'))
                    self.check_day += 1
                left.append(history_conversation_button)

            # The function called when any history_conversation_button is clicked
            def this_conversation(event):
                check = 0
                for obj in left.objects:
                    if isinstance(obj, pn.widgets.Button) and obj.button_type == 'success':
                        obj.button_type = 'primary'
                        check += 1

                if check == 0:
                    if self.history != []:

                        name = self.history[0]['User']
                        name = name.split()
                        name = ' '.join(name[:6])
                        history_conversation_button = pn.widgets.Button(name=f"{name}", button_type='primary', align='center', width=220, height=60)
                        history_conversation_button.history = self.history  # Store the history in the button
                        history_conversation_button.on_click(this_conversation)
                        if self.check_day == 0:  # Need to change this method to take into account the day
                            left.append(pn.pane.Markdown(f"<h2 style='color:white; font-size:18px;'>{self.day}</h2>", align='center'))
                            self.check_day += 1
                        left.append(history_conversation_button)
                        print('-------------self.history---------------')
                        print(self.history)
                        print('---------------------------------------')
                        self.history_history[0].append(self.history)
                        print('-------------self.history_history---------------')
                        print(self.history_history)
                        print('---------------------------------------')
                        if self.cookie_check:
                            self.mongo(self.history_history[0])
                            chiffre = 0
                            for conversation in collection.find({"user_id": uuid_v4}):
                                print(f'-------------{chiffre}---------------')
                                print(conversation)
                                chiffre += 1

                event.obj.button_type = 'success'
                button = event.obj
                old_chat = New_Chat_Interface()
                old_right = right_column(old_chat)

                # The length calculated to pop the too many chat interface created by the cookie method (layout.append(cookie))
                # length = 1
                # for obj in right.objects:
                    # length += 1
                
                layout.pop(1)
                if self.count_pop == 0:
                    self.count_pop += 1
                    layout.pop(1)
                
                layout.append(old_right)

                introduction(old_chat)

                list_of_conversation = button.history  # Access the history from the button
                self.history = button.history
                print('----------List Of Conversation-------------')
                print(list_of_conversation)
                print('-------------------------------------------')
                if list_of_conversation != []:
                    if not isinstance(list_of_conversation[0], dict):
                        list_of_conversation = list_of_conversation[0]
                        print('--------------List Of Conversation Check-----------------')
                        print(list_of_conversation)
                        print('---------------------------------------------------')
                for entry in list_of_conversation:
                    print('----------Entry-------------')
                    print(entry)
                    print('----------------------------')
                    if entry:
                        user = entry.get('User')
                        # document_used = entry.get('Document_used')
                        reservoirchat = entry.get('ReservoirChat')
                        old_chat.send(user,
                                    user="User",
                                    respond=False
                                    )
                        old_chat.send(reservoirchat,
                                    user="ReservoirChat",
                                    respond=False
                                    )

            # Not integrated yet
            def clear_history(event, chat):
                self.history = []
                introduction(chat)
            
            # Cookie popup with 2 buttons that will callbacollection.delete_many({}) ck to set_cookie_consent
            def cookie_consent_popup():
                if self.cookie_check:
                    return
                consent_text = pn.pane.Markdown(
                    "Hello, we use cookies to store your conversations and messages. By accepting, you agree to our use of cookies",
                    align="center"
                )
                accept_button = pn.widgets.Button(name='Accept', button_type='primary')
                decline_button = pn.widgets.Button(name='Decline', button_type='warning')

                accept_button.on_click(lambda event: set_cookie_consent('accepted'))
                decline_button.on_click(lambda event: set_cookie_consent('declined'))

                popup = pn.Column(
                    consent_text,
                    pn.Row(accept_button, decline_button, align="center"),
                    styles = {'background': 'lightgray'},
                    margin=(10, 10, 10, 10),
                    width=400,
                    align='center',
                    css_classes=['cookie-consent']
                )
                
                return popup
            
            # The Cookie Buttons Callback
            def set_cookie_consent(consent):
                if consent == 'accepted':
                    cookie_popup.visible = False
                    cookie = self.save_to_cookies("check", True)
                    layout.append(cookie)
                    print(pn.state.cookies.get("check"))
                    self.cookie_check = True
                elif consent == 'declined':
                    cookie_popup.visible = False

            new_conversation_button = pn.widgets.Button(name='New Conversation', button_type='primary', align='center')
            pn.bind(new_conversation, new_conversation_button, watch=True)
            
            cookie_popup = cookie_consent_popup()
            chat_interface = New_Chat_Interface()
            left = left_column()
            right = right_column(chat_interface)

            introduction(chat_interface)
            
            layout = layout(left, right)

            if self.cookie_check:
                if pn.state.cookies.get("uuid") and self.history_history != [[]]:
                    print('-------------self.history_history[0]---------------')
                    print(self.history_history[0])
                    print('---------------------------------------')
                    
                    # Iterate over the conversations in self.history_history[0]
                    for conversation in self.history_history[0]:
                        if conversation['interactions']:
                            print('-------------Conversation---------------')
                            print(conversation)
                            print('---------------------------------------')
                            
                            # Extract the first few words of the user's message to generate the button name
                            name = conversation['interactions'][0]['User']
                            name = name.split()
                            name = ' '.join(name[:6])  # Use the first six words of the user's message for the button name
                            
                            # Create the button
                            history_conversation_button = pn.widgets.Button(name=name, button_type='primary', align='center', width=220, height=60)
                            
                            # Store the individual interaction in the button's history
                            history_conversation_button.history = conversation['interactions']
                            
                            # Bind the button to the function
                            history_conversation_button.on_click(this_conversation)
                            
                            # Extract the date from the conversation and use it instead of self.day
                            conversation_date = conversation['date']
                            
                            # Check if the date has already been added
                            if self.check_day == 0:
                                left.append(pn.pane.Markdown(
                                    f"<h2 style='color:white; font-size:18px;'>{conversation_date}</h2>",
                                    align='center'
                                ))
                                self.check_day += 1
                            
                            # Append the button to the layout
                            left.append(history_conversation_button)

                    # Update self.history_history format to match the desired structure
                    self.history_history = [[[
                        {
                            'User': interaction['User'],
                            'ReservoirChat': interaction['ReservoirChat']
                        } for interaction in conversation['interactions']
                    ] for conversation in self.history_history[0]]]

            return layout

    # This is the instructions the LLM will follow
    system_message = '''You are ReservoirChat, a helpful, smart, kind, and efficient AI assistant. 
            You are specialized in reservoir computing.
            If a user ask a question in a language, you will respond in this language.
            You will serve as an interface to a RAG (Retrieval-Augmented Generation).
            When given documents, you will respond using only these documents. They will serve as inspiration to your response.
            If you are not given any document, you will exactly respond : "I didn't found any relevant information about on the documents I was provided, please refer to the [issue](https://github.com/VirgileBoraud/ReservoirChat/issues) of the github if it should be".
            You will never invent a source if it was not given.
            You will give the source if given.
            When asked to code, you will use the given additional code content,
            these documents are the only inspiration you can have to respond to the query of the user.
            You will not return the exact responses your were provided, but they will serve as inspiration for your real response.
            You will never use the database you have acquired elsewhere other than in the documents given.
            You will be given a list in this format:
            {"User": where the previous question of the user is,"Document_used": where the previous documents used are,"ReservoirChat": the response you used for the previous question, as the user can refer to them, you will take them into account for any question of the users.}
            Don't ever ever ever write the source when speaking to the user.
            You will use the following documents to respond to your task:
            DOCUMENTS:
            (document name : where the name of the document with the similarity are, you will display the name of these documents when you are asked to give the source | ticket: Where the similar questions or embeddings are | answer: the answer or resulting embeddings that will serve you as inspiration to your answer)
            '''

    if __name__ == "__main__":
        if not pn.state.cookies.get("uuid"):
            uuid_v4 = str(uuid.uuid4())
        else:
            uuid_v4 = pn.state.cookies.get("uuid")
        collection = db[uuid_v4]
        mangollection = ma[uuid_v4]
        reservoirchat = ReservoirChat(model_url='http://localhost:8000/v1',
                        embedding_url='http://127.0.0.1:5000/v1',
                        api_key='EMPTY',
                        embedding_model='nomic-ai/nomic-embed-text-v1.5',
                        model='TechxGenus/Codestral-22B-v0.1-GPTQ',
                        message=system_message,
                        similarity_threshold=0.6,
                        top_n=5)

        # file_list = reservoirchat.file_list('doc/md') # The file list used to load the new data into the RAG application
        # reservoirchat.load_data(file_list) # To load new data into the RAG application
        uuid_cookie = reservoirchat.save_to_cookies("uuid", uuid_v4)
        layout = reservoirchat.interface() # Creating the layout that will be returned to be called in the pn.serve() function
        layout.append(uuid_cookie)
        return layout

# Tiktoken has to download a file, on plafrim a tunnel must be set to access internet, so the file must be downloaded and put in a cache instead
tiktoken_cache_dir = "tiktoken_cache"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir

# validate
assert os.path.exists(os.path.join(tiktoken_cache_dir,"9b5ad71b2ce5302211f9c61530b329a4922fc6a4"))

# Serving the app in a bokeh server
# pn.serve(app, title="ReservoirChat", port=8080) # For local
# pn.serve(app, title="ReservoirChat", port=8080, address='localhost', allow_websocket_origin=['localhost:8080']) # For tests
pn.serve(app, title="ReservoirChat", port=8080, websocket_origin=["chat.reservoirpy.inria.fr"]) # For web