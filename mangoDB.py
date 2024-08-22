from pymongo import MongoClient

# Connect to the MongoDB server
client = MongoClient('localhost', 27017)

# Create or connect to a database
db = client['ReservoirChat']

# Create or connect to a collection
collection = db['35d286e5-687c-4708-821a-a1ff32a2c58a']

# Retrieve and print all documents in the collection
for item in collection.find():
    print(item)

# Close the connection
client.close()