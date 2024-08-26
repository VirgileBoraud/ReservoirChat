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

history_history = [[]]
for conversation in collection.find({"user_id": "840e50c3-157f-4fa3-8c97-6de90ce9333c"}):
    history_history[0].append(conversation)

print(history_history)
# Close the connection
client.close()