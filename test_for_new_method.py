from uuid import uuid4

import requests

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

import panel as pn

TEXT = "https://raw.githubusercontent.com/langchain-ai/langchain/master/docs/docs/modules/state_of_the_union.txt"

TEMPLATE = """Answer the question based only on the following context:

{context}

Question: {question}
"""

pn.extension(design="material")

prompt = ChatPromptTemplate.from_template(TEMPLATE)


@pn.cache
def get_vector_store():
    full_text = requests.get(TEXT).text
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(full_text)
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_texts(texts, embeddings)
    return db


db = get_vector_store()


def get_chain(callbacks):
    retriever = db.as_retriever(callbacks=callbacks)
    model = ChatOpenAI(callbacks=callbacks)

    def format_docs(docs):
        text = "\n\n".join([d.page_content for d in docs])
        return text

    def hack(docs):
        # https://github.com/langchain-ai/langchain/issues/7290
        for callback in callbacks:
            callback.on_retriever_end(docs, run_id=uuid4())
        return docs

    return (
        {"context": retriever | hack | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
    )


async def callback(contents, user, instance):
    callback_handler = pn.chat.langchain.PanelCallbackHandler(instance)
    chain = get_chain(callbacks=[callback_handler])
    response = await chain.ainvoke(contents)
    return response.content


pn.chat.ChatInterface(callback=callback).servable()