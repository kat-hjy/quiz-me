"""This module contains the TextIndexing class for indexing text documents."""

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader  # , PyPDFLoader
from loguru import logger
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores.base import VectorStoreRetriever
import chromadb
from chromadb import ClientAPI
from langchain_chroma import Chroma
from chromadb.api.models import Collection
from langchain_core.embeddings.embeddings import Embeddings
from langchain_voyageai import VoyageAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()


class TextIndexing:
    """TextIndexing class for indexing text documents."""

    def __init__(self, vectorstore_dir: str, collection_name: str):
        """Initialize the Indexing class.

        Args:
            vectorstore_dir (str): Directory to store the vectorstore.
            collection_name (str): Name of the collection to store documents to.
        """
        VOYAGE_MODEL = os.getenv("VOYAGE_MODEL")
        self.persistent_client: ClientAPI = chromadb.PersistentClient(
            path=vectorstore_dir
        )
        self.collection: Collection = self.persistent_client.get_or_create_collection(
            collection_name
        )
        self.embeddings: Embeddings = VoyageAIEmbeddings(model=VOYAGE_MODEL)
        self.vectorstore: Chroma = Chroma(
            client=self.persistent_client,
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=vectorstore_dir,
            create_collection_if_not_exists=False,
        )
        self.retriever: VectorStoreRetriever = None

    def load_documents(self, dir_path: str) -> list[Document]:
        """Load PDF documents from a directory.

        Args:
            dir_path (str): Path to the directory containing PDF documents.

        Returns:
            list[Document]: List of loaded documents.
        """
        # TODO: find out how to check if all documents are .pdf files
        # Loader for multiple documents
        multi_loader: PyPDFDirectoryLoader = PyPDFDirectoryLoader(
            path=dir_path, recursive=True, extract_images=False
        )
        multi_docs: list[Document] = multi_loader.load()
        logger.debug(f"Loaded {len(multi_docs)} documents.")
        logger.debug("Checking contents of first document:")
        for key, value in dict(multi_docs[0]).items():
            logger.debug(f"{key}: {value}")
        return multi_docs

    def preprocess_documents(self, docs: list[Document]) -> list[Document]:
        """Clean the documents by:
        | replacing nextline characters with white spaces, and
        | removing non-ascii characters and repeated whitespace characters.

        Args:
            docs (list[Document]): list of documents.

        Returns:
            list[Document]: list of cleaned documents.
        """
        for doc in docs:
            # replace \n with " " in documents
            doc.page_content = doc.page_content.replace("\n", " ")
            # remove all non-ascii characters
            doc.page_content = "".join(i for i in doc.page_content if ord(i) < 128)
            # remove repeated whitespace characters
            doc.page_content = " ".join(doc.page_content.split())
        return docs

    def split_documents(self, docs: list[Document]) -> list[Document]:
        """Split the documents into smaller chunks.

        Args:
            docs (list[Document]): list of documents.

        Returns:
            list[Document]: list of smaller documents.
        """
        text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=0
        )
        splits: list[Document] = text_splitter.split_documents(docs)
        logger.debug(f"Split {len(docs)} documents into {len(splits)} chunks.")
        return splits

    def store_documents(
        self, vectorstore_dir: str, collection_name: str, docs: list[Document]
    ) -> VectorStoreRetriever:
        """Initialize and store documents into the vectorstore.

        Args:
            vectorstore_dir (str): Directory to store the vectorstore.
            collection_name (str): Name of the collection to store documents to.
            docs (list[Document]): list of documents to store.

        Returns:
            VectorStoreRetriever: VectorStoreRetriever initialized from this VectorStore.
        """
        self.vectorstore: Chroma = Chroma.from_documents(
            client=self.persistent_client,
            collection_name=collection_name,
            embedding=self.embeddings,
            persist_directory=vectorstore_dir,
            documents=docs,
        )
        logger.debug(self.collection.peek())
        return self.vectorstore

    def get_vectorstore(self) -> Chroma:
        """Return the vectorstore.

        Returns:
            Chroma: VectorStore.
        """
        return self.vectorstore

    def get_retriever(self, vectorstore: Chroma) -> VectorStoreRetriever:
        """Return VectorStoreRetriever initialized from the VectorStore.

        Args:
            vectorstore (Chroma): VectorStore to initialize the retriever from.

        Returns:
            VectorStoreRetriever: VectorStoreRetriever initialized from this VectorStore.
        """
        self.retriever: VectorStoreRetriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 6}
        )
