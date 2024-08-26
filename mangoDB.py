from pymongo import MongoClient
import uuid
import time
from datetime import datetime

# Connect to the MongoDB server
client = MongoClient('localhost', 27017)

# Create or connect to a database
db = client['ReservoirChat']

# Create or connect to a collection
collection = db['conversations']

# Function to format the date as per your custom method
def get_custom_date():
    current_time = time.ctime()
    time_components = current_time.split()
    custom_date = " ".join(time_components[0:3])
    return custom_date

# Function to insert or update conversations from self.history_history
def update_conversations(history_history, user_id):
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
            "user_id": user_id,  # User-specific ID
            "date": get_custom_date(),  # Custom formatted date
            "interactions": interactions_with_timestamps
        }
        
        # Insert the conversation document into the collection
        collection.insert_one(conversation_doc)

# Example usage
# Assuming self.history_history is already populated
self_history_history = [
    [{"User": "How does this work?", "ReservoirChat": "This is how it works..."}, {"User": "Can you explain further?", "ReservoirChat": "Sure, here are more details..."}],
    [{"User": "What are your hours?", "ReservoirChat": "We are open 24/7."}, {"User": "Thank you!", "ReservoirChat": "You're welcome!"}]
]

user_id = "user123"  # Example user ID

# Update the conversations in the MongoDB collection
update_conversations(self_history_history, user_id)

self_history_history = [
    [{"User": "How does this work?", "ReservoirChat": "This is how it works..."}, {"User": "Can you explain further?", "ReservoirChat": "Sure, here are more details..."}],
    [{"User": "What are your hours?", "ReservoirChat": "We are open 24/7."}, {"User": "Thank you!", "ReservoirChat": "You're welcome!"}]
]
update_conversations(self_history_history, user_id)
# Retrieve and print all conversations for this user in the collection
for conversation in collection.find({"user_id": user_id}):
    print(conversation)

# Close the connection
client.close()