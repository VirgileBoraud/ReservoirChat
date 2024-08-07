from pymongo import MongoClient

# Connect to the MongoDB server
client = MongoClient('localhost', 27017)

# Create or connect to a database
db = client['mydatabase']

# Create or connect to a collection
collection = db['mycollection']

# Define a list of items to store in the collection
items = [
    {"item": "Apple", "quantity": 10},
    {"item": "Banana", "quantity": 5},
    {"item": "Orange", "quantity": 7},
    {"item": "Mango", "quantity": 3},
    {"item": "Grapes", "quantity": 12}
]

# Insert the list of items into the collection
collection.insert_many(items)

# Retrieve and print all documents in the collection
for item in collection.find():
    print(item)

# Close the connection
client.close()
