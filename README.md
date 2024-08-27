# ReservoirChat

ReservoirChat is a specialized AI model **powered by a Large Language Model (LLM)**. It uses a technique called **Retrieval-Augmented Generation (RAG)** to provide *accurate and hallucination-free informations*. It focuses on reservoir computing, and will use the reservoirPy library to code reservoirs. Reservoirs are a pool of randomly connected artificial neurons where, unlike the traditional neuron layers, only the readout layer, which is the last layer, is trained. It thus reduced the effective computational cost.

Unlike traditional LLMs, ReservoirChat integrates retrieval mechanisms and code, related to reservoir computing and the python library [reservoirPy](https://reservoirpy.readthedocs.io/en/latest/). This means it can deliver precise and reliable responses based on external data sources while leveraging large language model's capabilities.

It is important to note that the AI model is **powered by a LLM**, not a reservoir computing neuron network.

def mongo(self, data):
        document_name = "user_id" + str(uuid_v4)  # Create a unique document name using uuid
        if not self.collection.find_one({"_id": document_name}):
            # Insert the conversation into MongoDB with the document name as the _id field
            result = self.collection.insert_one({"user_id": uuid_v4, "history": data})
            print(f"Inserted document ID: {result.inserted_id}")
        else:
            # Update the existing conversation in MongoDB with new data
            result = self.collection.update_one({"_id": document_name}, {"$set": {"history": data}})
            print(f"Updated {result.modified_count} documents(s)")