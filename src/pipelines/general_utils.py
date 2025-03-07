"""This module contains general utility functions for the pipelines."""

import tiktoken
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from dotenv import load_dotenv
from loguru import logger
import os
import yaml
from typing import Dict, List
from langchain_core.documents import Document
from langgraph.pregel.io import AddableValuesDict
from datetime import datetime
import pytz

load_dotenv()


def save_output(output: Dict | AddableValuesDict, output_dir: str) -> None:
    """Saves the output to a txt file.

    Args:
        output (Dict | AddableValuesDict): The output to save.

    Returns:
        None
    """
    # Get Singapore timezone
    tz = pytz.timezone("Asia/Singapore")
    timestamp = datetime.now(tz).strftime("%Y%m%d_%H%M%S")

    # Create save path with timestamp
    save_path = os.path.join(output_dir, f"output_{timestamp}.txt")

    # Make output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save output to file
    with open(save_path, "w") as file:
        file.write(str(output))
    logger.info(f"Output saved to {save_path}.")


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string.

    Args:
        string (str): The text string.
        encoding_name (str): The name of the encoding to use. Defaults to "cl100k_base".

    Returns:
        int: The number of tokens in the text string.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def load_catalog() -> Dict:
    """Loads the catalog file and returns its contents.

    Returns:
        Dict: The contents of the catalog file.
    """
    CATALOG_PATH: str = os.getenv("CATALOG_PATH")

    with open(CATALOG_PATH, "r") as file:
        catalog: Dict = yaml.safe_load(file)
    # logger.debug(f"Catalog:\n{catalog}")
    return catalog


def load_config() -> dict:
    """Loads the config file and returns its contents.

    Returns:
        dict: The contents of the catalog file.
    """
    CONFIG_PATH: str = os.getenv("CONFIG_PATH")

    with open(CONFIG_PATH, "r") as file:
        config: dict = yaml.safe_load(file)
    # logger.debug(f"Config:\n{config}")
    return config


def print_docs(docs: List[Document]) -> None:
    """Prints the contents of the documents.

    Args:
        docs (List[Document]): The list of documents.
    """
    for i, doc in enumerate(docs):
        logger.info(f"Doc {i}: {doc.id}")
        logger.info(f"Page_content:\n{doc.page_content.replace('.', '.\n')}")
        logger.info("\nMetadata:")
        for key, value in doc.metadata.items():
            print(f"{key}: {value}")
        logger.info("=" * 50)


def read_template(template_path: str) -> str:
    """Reads the template file and returns its contents.

    Args:
        template_path (str): The path to the template file.

    Returns:
        str: The contents of the template file.
    """
    with open(template_path, "r") as file:
        template: str = file.read()
    return template


def get_progress_bar() -> Progress:
    """Creates and returns a progress bar.

    Returns:
        Progress: The progress bar.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
    )
