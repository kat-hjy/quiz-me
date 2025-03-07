"""This module contains the MultiModalIndexing class for indexing multimodal data."""

from unstructured.partition.pdf import partition_pdf
from loguru import logger
import os
import uuid
from langchain_chroma import Chroma
from pathlib import Path

from src.pipelines import general_utils as gu
from dotenv import load_dotenv
import chromadb
from chromadb import ClientAPI
from chromadb.api.models import Collection
from langchain_core.embeddings.embeddings import Embeddings

# from langchain_voyageai import VoyageAIEmbeddings
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_core.vectorstores.base import VectorStoreRetriever

load_dotenv()

# TODO: postprocess text and images to remove unwanted images and texts (e.g. NUS, or timetable/admin info)


class MultiModalIndexing:
    def __init__(self, vectorstore_dir: str, collection_name: str):
        """Initialize the MultiModalIndexing class.

        Args:
            vectorstore_dir (str): Directory to store the vectorstore.
            collection_name (str): Name of the collection to store documents to.
        """
        self.persistent_client: ClientAPI = chromadb.PersistentClient(
            path=vectorstore_dir
        )
        self.collection: Collection = self.persistent_client.get_or_create_collection(
            collection_name
        )
        # VOYAGE_MULTIMODAL_MODEL = os.getenv("VOYAGE_MULTIMODAL_MODEL")
        # self.embeddings: Embeddings = VoyageAIEmbeddings(model=VOYAGE_MULTIMODAL_MODEL)

        OPENCLIP_MODEL_NAME = os.getenv("OPENCLIP_MODEL_NAME")
        OPENCLIP_CHECKPOINT = os.getenv("OPENCLIP_CHECKPOINT")
        # OpenCLIP model
        self.embeddings: Embeddings = OpenCLIPEmbeddings(
            model_name=OPENCLIP_MODEL_NAME, checkpoint=OPENCLIP_CHECKPOINT
        )

        self.vectorstore: Chroma = Chroma(
            client=self.persistent_client,
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=vectorstore_dir,
            create_collection_if_not_exists=False,
        )
        self.retriever: VectorStoreRetriever = None

    def _process_pdf(
        self, path: str, figures_path: str
    ) -> tuple[list[str], list[str], list[str]]:
        """Process a PDF file.

        Args:
            path (str): Path to the PDF file.
            figures_path (str): Main directory to store extracted images.

        Returns:
            tuple[list[str], list[str], list[str]]: A tuple containing the texts, tables, and image URIs extracted from the
        """
        # Create unique subfolder for extracted figures based on PDF path
        pdf_path = Path(path)
        subfolder = "_".join(
            [pdf_path.parent.parent.stem, pdf_path.parent.stem, pdf_path.stem]
        )
        figures_path = Path(figures_path) / subfolder
        figures_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Processing PDF: {path}")
        # Extract images, tables, and chunk text
        logger.debug(f"image_path: {figures_path}")
        raw_pdf_elements = partition_pdf(
            filename=str(path),
            extract_image_block_types=["Image", "Table"],
            infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=1000,
            new_after_n_chars=800,
            combine_text_under_n_chars=200,
            extract_image_block_output_dir=str(figures_path),
        )
        logger.info(f"Extracted {len(raw_pdf_elements)} elements from PDF.")

        # Categorize text elements by type
        tables = []
        texts = []
        for element in raw_pdf_elements:
            if "unstructured.documents.elements.Table" in str(type(element)):
                tables.append(str(element))
            elif "unstructured.documents.elements.CompositeElement" in str(
                type(element)
            ):
                texts.append(str(element))

        # Get image URIs with .jpg extension only
        image_uris = sorted(
            [
                os.path.join(figures_path, image_name)
                for image_name in os.listdir(os.path.join(figures_path))
                if image_name.endswith(".jpg")
            ]
        )
        logger.debug(f"Extracted {len(image_uris)} images from PDF.")

        return texts, tables, image_uris

    def get_vectorstore(self) -> Chroma:
        """Get the vectorstore.

        Returns:
            Chroma: The vectorstore.
        """
        return self.vectorstore

    def get_retriever(self, vectorstore: Chroma) -> VectorStoreRetriever:
        """Get the retriever.

        Returns:
            VectorStoreRetriever: _description_
        """
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 6}
        )
        return self.retriever

    def process_pdf_dir(
        self, path: str, figures_path: str
    ) -> tuple[list[str], list[str], list[str]]:
        """Process a directory of PDFs.

        Args:
            path (str): Path to the directory of PDFs.
            figures_path (str): Main directory to store extracted images.

        Returns:
            tuple[list[str], list[str], list[str]]: A tuple containing the texts, tables, and image URIs extracted from the
        """
        # TODO: use multiprocessing as there was some pickle error with both
        # multiprocessing and concurrent.futures.ProcessPoolExecutor
        progress = gu.get_progress_bar()
        texts_combined = []
        tables_combined = []
        image_uris_combined = []
        logger.info(f"Processing PDFs in directory: {path}")
        logger.debug(f"figures_path: {figures_path}")
        with progress:
            pdfs = [pdf for pdf in Path(path).rglob("*.pdf") if pdf.is_file()]
            task = progress.add_task(
                f"[green]Processing {len(pdfs)} PDFs in directory...", total=len(pdfs)
            )

            for pdf_path in pdfs:
                try:
                    # Process the PDF
                    texts, tables, image_uris = self._process_pdf(
                        pdf_path, figures_path
                    )
                    logger.info(
                        f"Extracted {len(texts)} texts, {len(tables)} tables, and {len(image_uris)} images from PDF: {pdf_path}"
                    )

                    # TODO: remove similar images

                    # for debugging purposes
                    texts_combined.extend(texts)
                    tables_combined.extend(tables)
                    image_uris_combined.extend(image_uris)

                    # get the vectorstore
                    vectorstore = self.get_vectorstore()

                    # add images and texts to the vectorstore
                    self.add_images_texts_to_vectorstore(vectorstore, texts, image_uris)

                    # update the progress bar
                    progress.update(task, advance=1)
                except Exception as e:
                    logger.error(f"Error processing PDF {pdf_path}: {e}", exc=e)

        return texts_combined, tables_combined, image_uris_combined

    def add_images_texts_to_vectorstore(
        self, vectorstore: Chroma, texts: list[str], image_uris: list[str]
    ) -> None:
        """Add images and texts to the vectorstore.

        Args:
            texts (list[str]): List of texts to add to the vectorstore.
            image_uris (list[str]): List of image URIs to add to the vectorstore.

        Returns:
            None
        """
        logger.info("Adding images and texts to the vectorstore...")
        if image_uris:
            # Generate unique IDs for images
            image_ids = [str(uuid.uuid4()) for _ in image_uris]
            # Add images with IDs
            vectorstore.add_images(uris=image_uris, ids=image_ids)

        if texts:
            # Generate unique IDs for texts
            text_ids = [str(uuid.uuid4()) for _ in texts]
            # Add documents with IDs
            vectorstore.add_texts(texts=texts, ids=text_ids)

        del image_uris, texts

        logger.info("Added images and texts to the vectorstore.")
