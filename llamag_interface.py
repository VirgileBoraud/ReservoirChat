import panel as pn
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from openai import OpenAI

class LLaMag:
    def __init__(self, base_url, api_key, model="nomic-ai/nomic-embed-text-v1.5-GGUF", similarity_threshold=75, top_n=5):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
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

    def load_data(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                data = file.read()

            questions_answers = data.split("Question: ")
            qa_pairs = []
            for qa in questions_answers[1:]:
                parts = qa.split("Answer: ")
                question = parts[0].strip()
                answer = parts[1].strip() if len(parts) > 1 else ""
                qa_pairs.append({"question": question, "answer": answer})

            self.df = pd.DataFrame(qa_pairs)
            self.df['question_embedding'] = self.df['question'].apply(lambda x: self.get_embedding(x))
            self.df.to_csv('qa_embeddings.csv', index=False)
        except Exception as e:
            print(f"Error loading data: {e}")

    def find_most_similar_question(self, query):
        try:
            query_embedding = self.get_embedding(query)
            self.df['similarity'] = self.df['question_embedding'].apply(lambda x: 1 - cosine(query_embedding, x))
            most_similar_idx = self.df['similarity'].idxmax()
            most_similar_qa = self.df.iloc[most_similar_idx]
            similarity_percentage = self.df['similarity'].iloc[most_similar_idx] * 100
            return most_similar_qa, similarity_percentage
        except Exception as e:
            print(f"Error finding most similar question: {e}")
            return None, 0

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
            full_prompt = f"{context}\n\n{prompt}"
            response = self.client.completions.create(
                model=self.model,
                prompt=full_prompt,
                max_tokens=500,
                temperature=0.5
            )
            print(f"LLM Response: {response}")
            return response.choices[0].text.strip()
        except Exception as e:
            print(f"Error getting LLM answer: {e}")
            return "Error generating answer from LLM."

    def get_answer(self, query):
        most_similar_qa, similarity_percentage = self.find_most_similar_question(query)
        if self.is_coding_request(query):
            context = "\n".join([f"Q: {row['question']}\nA: {row['answer']}" for idx, row in self.df.iterrows()])
            return self.get_llm_answer(query, context), similarity_percentage, []
        elif similarity_percentage >= self.similarity_threshold:
            similar_responses = self.find_top_similar_questions(query)
            return most_similar_qa['answer'], similarity_percentage, similar_responses
        else:
            context = "\n".join([f"Q: {row['question']}\nA: {row['answer']}" for idx, row in self.df.iterrows()])
            return self.get_llm_answer(query, context), similarity_percentage, []

    def respond(self, query):
        answer, similarity_percentage, similar_responses = self.get_answer(query)
        response = f"Similarity: {similarity_percentage:.2f}%\nQuery: {query}\nAnswer: {answer}"
        if isinstance(similar_responses, pd.DataFrame) and not similar_responses.empty:
            response += "\n\nTop Similar Responses:"
            for index, sim_response in similar_responses.iterrows():
                response += f"\nSimilarity: {sim_response['similarity_percentage']:.2f}%\nQuestion: {sim_response['question']}\nAnswer: {sim_response['answer']}\n"
        return response

    def interactive_query(self, query):
        response = self.respond(query)
        return pn.pane.Markdown(response)

    def interface(self):
        input_query = pn.widgets.TextInput(name='Enter your query:')
        output = pn.bind(self.interactive_query, query=input_query)

        layout = pn.Column(
            pn.pane.Markdown("# LLaMag"),
            input_query,
            output
        )

        pn.serve(layout, start=True)

llama_rag = LLaMag(base_url="http://localhost:1234/v1", api_key="lm-studio", top_n=5)
llama_rag.load_data('doc/Q&A_format.md')
llama_rag.interface()