"""This module contains the MultiModalIndexing class for indexing multimodal documents."""

from langchain_core.documents import Document
from loguru import logger
import chromadb
from chromadb import ClientAPI
from langchain_chroma import Chroma
from chromadb.api.models import Collection
from langchain_core.embeddings.embeddings import Embeddings
from langchain_voyageai import VoyageAIEmbeddings
from dotenv import load_dotenv
import os
from pathlib import Path
import quiz_me.pipelines.general_utils as gu
from langchain_text_splitters import CharacterTextSplitter
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Element
import time
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
import base64
from langchain_core.messages import HumanMessage
import uuid

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage.file_system import LocalFileStore


load_dotenv()


class MultiModalIndexing:
    def __init__(
        self,
        docstore_dir: str,
        vectorstore_dir: str,
        collection_name: str,
        llm: ChatAnthropic,
    ):
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
        self.retriever: MultiVectorRetriever = None
        self.llm = llm
        # Initialize the storage layer
        self.store: LocalFileStore = LocalFileStore(root_path=docstore_dir)
        self.id_key = "doc_id"

    def load_documents(self, dir_path: str, output_dir: str) -> None:
        """Load PDF documents from a directory.

        Args:
            dir_path (str): Path to the directory containing PDF documents.
            output_dir (str): Path to save extracted images
        """
        progress = gu.get_progress_bar()
        pdf_files = list(Path(dir_path).rglob("*.pdf"))
        task = progress.add_task("Processing PDFs", total=len(pdf_files))

        with progress:
            for pdf in pdf_files:
                # Create subdirectory path using pdf file hierarchy
                pdf_path_obj = Path(pdf)
                subfolder = "_".join(
                    [
                        pdf_path_obj.parent.parent.stem,
                        pdf_path_obj.parent.stem,
                        pdf_path_obj.stem,
                    ]
                )
                save_path = Path(output_dir) / subfolder

                # Ensure directory exists
                save_path.mkdir(parents=True, exist_ok=True)

                # Convert to absolute path to ensure unstructured uses correct location
                abs_save_path = str(save_path.resolve())
                pdf = str(pdf.resolve())
                logger.debug(f"Saving images to: {abs_save_path}")
                logger.info(f"Processing {pdf}")
                # Get elements
                substeps = [
                    "Extracting elements",
                    "Categorizing elements",
                    "Splitting text into 4k token chunks",
                    "Generating text and table summaries",
                    "Generating image summaries",
                    "Indexing documents",
                ]
                inner_task = progress.add_task(
                    "[cyan]Processing steps", total=len(substeps)
                )

                progress.update(
                    inner_task, advance=1, description=f"[cyan]{substeps[0]}"
                )

                raw_pdf_elements = self._extract_pdf_elements(pdf, abs_save_path)

                # Get text, tables
                progress.update(
                    inner_task, advance=1, description=f"[cyan]{substeps[1]}"
                )
                texts, tables = self._categorize_elements(raw_pdf_elements)

                # Optional: Enforce a specific token size for texts
                progress.update(
                    inner_task, advance=1, description=f"[cyan]{substeps[2]}"
                )
                text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
                    chunk_size=4000, chunk_overlap=0
                )
                joined_texts = " ".join(texts)
                texts_4k_token = text_splitter.split_text(joined_texts)

                # Use multi-vector-retriever to index image (and / or text, table) summaries,
                # but retrieve raw images (along with raw texts or tables).
                progress.update(
                    inner_task, advance=1, description=f"[cyan]{substeps[3]}"
                )
                text_summaries, table_summaries = self._generate_text_summaries(
                    texts_4k_token, tables, summarize_texts=True
                )

                # Image summaries
                progress.update(
                    inner_task, advance=1, description=f"[cyan]{substeps[4]}"
                )
                img_base64_list, image_summaries = self._generate_img_summaries(
                    abs_save_path
                )

                # Add to vectorstore
                # Add raw docs and doc summaries to Multi Vector Retriever:
                # Store the raw texts, tables, and images in the docstore.
                # Store the texts, table summaries, and image summaries in the vectorstore for efficient semantic retrieval.
                # Create retriever
                progress.update(
                    inner_task, advance=1, description=f"[cyan]{substeps[5]}"
                )
                self._index_documents(
                    self.vectorstore,
                    text_summaries,
                    texts,
                    table_summaries,
                    tables,
                    image_summaries,
                    img_base64_list,
                )
                progress.advance(task_id=task, advance=1)

        logger.info("Finished processing PDFs")

    def _index_documents(
        self,
        vectorstore: Chroma,
        text_summaries: list[str],
        texts: list[str],
        table_summaries: list[str],
        tables: list[str],
        image_summaries: list[str],
        images: list[str],
    ) -> None:
        """
        Uses a retriever that indexes summaries, but returns raw images or texts to index documents.

        Args:
            vectorstore: Chroma
            text_summaries: List of text summaries
            texts: List of texts
            table_summaries: List of table summaries
            tables: List of tables
            image_summaries: List of image summaries
            images: List of images
        """
        # Create the multi-vector retriever
        self.retriever: MultiVectorRetriever = self.get_retriever(vectorstore)

        # Helper function to add documents to the vectorstore and docstore
        def add_documents(retriever, doc_summaries, doc_contents):
            doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
            summary_docs = [
                Document(page_content=s, metadata={self.id_key: doc_ids[i]})
                for i, s in enumerate(doc_summaries)
            ]
            retriever.vectorstore.add_documents(summary_docs)
            # Convert strings to bytes objects before storing
            doc_contents_bytes = [
                dc.encode("utf-8") if isinstance(dc, str) else dc for dc in doc_contents
            ]
            retriever.docstore.mset(list(zip(doc_ids, doc_contents_bytes)))

        # Add texts, tables, and images
        # Check that text_summaries is not empty before adding
        if text_summaries:
            logger.debug("Adding text summaries to the vectorstore")
            add_documents(self.retriever, text_summaries, texts)
        # Check that table_summaries is not empty before adding
        if table_summaries:
            logger.debug("Adding table summaries to the vectorstore")
            add_documents(self.retriever, table_summaries, tables)
        # Check that image_summaries is not empty before adding
        if image_summaries:
            logger.debug("Adding image summaries to the vectorstore")
            add_documents(self.retriever, image_summaries, images)

    # Extract elements from PDF
    def _extract_pdf_elements(self, pdf_path: str, output_dir: str) -> list[Element]:
        """
        Extract images, tables, and chunk text from a PDF file.
        Args:
            pdf_path: path of .pdf file
            output_dir: path to save extracted images

        Returns:
            list[Element]: List of elements extracted from the PDF
        """
        logger.debug(f"pdf_path: {pdf_path}")
        logger.debug(f"output_dir: {output_dir}")

        start = time.time()
        results = partition_pdf(
            filename=str(pdf_path),
            extract_images_in_pdf=True,
            # infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=4000,
            new_after_n_chars=3800,
            combine_text_under_n_chars=2000,
            extract_image_block_output_dir=str(output_dir),  # Use absolute path
            extract_image_block_types=["Image", "Table"],  # Explicitly specify types
        )
        end = time.time()

        logger.debug(f"Time taken to extract elements: {end - start} seconds")
        logger.debug(f"Saved images to: {output_dir}")
        return results

    # Categorize elements by type
    def _categorize_elements(
        self, raw_pdf_elements: list[Element]
    ) -> tuple[list[str], list[str]]:
        """
        Categorize extracted elements from a PDF into tables and texts.

        Args:
            raw_pdf_elements: List of unstructured.documents.elements

        Returns:
            tuple[list[str], list[str]]: texts, tables
        """
        tables = []
        texts = []
        for element in raw_pdf_elements:
            if "unstructured.documents.elements.Table" in str(type(element)):
                tables.append(str(element))
            elif "unstructured.documents.elements.CompositeElement" in str(
                type(element)
            ):
                texts.append(str(element))
        return texts, tables

    # Generate summaries of text elements
    def _generate_text_summaries(
        self, texts: list[str], tables: list[str], summarize_texts: bool = False
    ) -> tuple[list[str], list[str]]:
        """
        Summarize text elements
        Args:
            texts(list[str]): List of text elements
            tables(list[str]): List of table elements
            summarize_texts(bool): Whether to summarize text elements

        Returns:
            tuple[list[str], list[str]]: text_summaries, table_summaries
        """

        # Prompt
        prompt_text = """You are an assistant tasked with summarizing tables and text for retrieval. \
        These summaries will be embedded and used to retrieve the raw text or table elements. \
        Give a concise summary of the table or text that is well optimized for retrieval. Table or text: {element} """
        prompt = ChatPromptTemplate.from_template(prompt_text)

        # Text summary chain
        summarize_chain = (
            {"element": lambda x: x} | prompt | self.llm | StrOutputParser()
        )

        # Initialize empty summaries
        text_summaries = []
        table_summaries = []

        # Apply to text if texts are provided and summarization is requested
        if texts and summarize_texts:
            text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})
        elif texts:
            text_summaries = texts

        # Apply to tables if tables are provided
        if tables:
            table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})

        return text_summaries, table_summaries

    def _encode_image(self, image_path: str) -> str:
        """Getting the base64 string

        Args:
            image_path (str): image path

        Returns:
            str: base64 representation of the image
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _image_summarize(self, img_base64, prompt):
        """Make image summary"""

        msg = self.llm.invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            },
                        },
                    ]
                )
            ]
        )
        return msg.content

    def _generate_img_summaries(self, path: str) -> tuple[list[str], list[str]]:
        """
        Generate summaries and base64 encoded strings for images

        Args:
            path: Path to list of .jpg files extracted by Unstructured

        Returns:
            img_base64_list: List of base64 encoded images
            image_summaries: List of image summaries
        """

        # Store base64 encoded images
        img_base64_list = []

        # Store image summaries
        image_summaries = []

        # Prompt
        prompt = """You are an assistant tasked with summarizing images for retrieval. \
        These summaries will be embedded and used to retrieve the raw image. \
        Give a concise summary of the image that is well optimized for retrieval."""

        logger.debug(f"Generating image summaries for {path}")
        # Apply to images
        for img_file in sorted(os.listdir(path)):
            if img_file.endswith(".jpg"):
                img_path = os.path.join(path, img_file)
                logger.debug(f"img_file: {img_file}")
                logger.debug(f"img_path: {img_path}")
                base64_image = self._encode_image(img_path)
                img_base64_list.append(base64_image)
                image_summaries.append(self._image_summarize(base64_image, prompt))

        return img_base64_list, image_summaries

    def get_retriever(self, vectorstore: Chroma) -> MultiVectorRetriever:
        """Get a MultiVectorRetriever instance.

        Args:
            vectorstore (Chroma): The vectorstore to use for retrieval.

        Returns:
            MultiVectorRetriever: A MultiVectorRetriever instance.
        """
        self.retriever: MultiVectorRetriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=self.store,
            id_key=self.id_key,
        )
        # Test output
        logger.debug(f"Retriever: {self.retriever}")
        logger.debug(
            f"Test query: {self.retriever.invoke('digestive system', limit=6)}"
        )
        return self.retriever

    def get_vectorstore(self) -> Chroma:
        """Get the vectorstore.

        Returns:
            Chroma: The vectorstore.
        """
        return self.vectorstore
