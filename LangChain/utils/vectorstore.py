from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from config import Config

config = Config()

def create_vectorstore(chunks):
    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def load_vectorstore(path: str):
    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    return vectorstore