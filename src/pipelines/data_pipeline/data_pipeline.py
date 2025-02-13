import os
from langchain.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

class DataPipeline():
    # Load lecture notes
    def load_documents(folder_path):
        docs = []
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            else:
                loader = UnstructuredFileLoader(file_path)
            docs.extend(loader.load())
        return docs

    # Split text into chunks
    def split_text(docs):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        return text_splitter.split_documents(docs)

    # Create embeddings and store in FAISS
    def create_vector_db(docs, db_path="faiss_index"):
        embeddings = OpenAIEmbeddings()
        vector_db = FAISS.from_documents(docs, embeddings)
        vector_db.save_local(db_path)
        return vector_db
