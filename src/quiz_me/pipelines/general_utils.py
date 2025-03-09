"""This module contains general utility functions for the pipelines."""

import tiktoken
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from dotenv import load_dotenv
from loguru import logger
import os
import yaml
from typing import Dict, List
from langchain_core.documents import Document
from datetime import datetime
import pytz
import json
from langgraph.pregel.io import AddableValuesDict

load_dotenv()


def serialize_document(doc: Document) -> dict:
    """Serialize a Document object to a dictionary."""
    return {"page_content": doc.page_content, "metadata": doc.metadata}


def serialize_state(state: AddableValuesDict) -> dict:
    """Serialize a AddableValuesDict object to a dictionary."""
    return {k: serialize_value(v) for k, v in state.items()}


def serialize_value(value: any) -> any:
    """Recursively serialize values to JSON-compatible format."""
    if isinstance(value, Document):
        return serialize_document(value)
    elif isinstance(value, list):
        return [serialize_value(item) for item in value]
    elif isinstance(value, dict):
        return {k: serialize_value(v) for k, v in value.items()}
    elif isinstance(value, AddableValuesDict):
        return serialize_state(value)
    elif hasattr(value, "dict"):  # For Pydantic models
        return value.dict()
    elif hasattr(value, "to_dict"):  # For other objects with to_dict method
        return value.to_dict()
    return value


def save_output(output: Dict | AddableValuesDict, output_dir: str) -> tuple[bool, any]:
    """Save the output to a JSON file with timestamp.

    Args:
        output: The output to save
        output_dir: Directory to save the output file

    Returns:
        bool: True if output is saved successfully as .json, False otherwise
        any: The serialized output
    """
    # Get Singapore timezone
    tz = pytz.timezone("Asia/Singapore")
    timestamp = datetime.now(tz).strftime("%Y%m%d_%H%M%S")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create save path with timestamp
    save_path = os.path.join(output_dir, f"output_{timestamp}.json")

    try:
        # Serialize the output
        serialized_output = serialize_value(output)

        # Save to JSON file
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(serialized_output, f, indent=2, ensure_ascii=False)
        logger.info(f"Output saved to {save_path}")
        return True, serialized_output
    except Exception as e:
        logger.error(f"Error saving output: {e}")
        # Save as text if JSON serialization fails
        txt_path = os.path.join(output_dir, f"output_{timestamp}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(str(output))
        logger.warning(f"Saved output as text file instead: {txt_path}")
        return False, output


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

    Args:
        catalog_path (str): The path to the catalog file. Defaults to None.
    Returns:
        Dict: The contents of the catalog file.
    """
    CATALOG_PATH: str = os.getenv("CATALOG_PATH")

    with open(CATALOG_PATH, "r") as file:
        catalog: Dict = yaml.safe_load(file)
    # logger.debug(f"Catalog:\n{catalog}")
    return catalog


def load_config(config_path: str = None) -> dict:
    """Loads the config file and returns its contents.

    Args:
        config_path (str): The path to the config file. Defaults to None.
    Returns:
        dict: The contents of the catalog file.
    """
    CONFIG_PATH = config_path or os.getenv("CONFIG_PATH")
    if not CONFIG_PATH:
        raise ValueError("CONFIG_PATH not provided.")
    with open(CONFIG_PATH, "r") as file:
        config: dict = yaml.safe_load(file)
    # logger.debug(f"Config:\n{config}")
    return config


def print_docs(docs: List[Document]) -> None:
    """Prints the contents of the documents.

    Args:
        docs (List[Document]): The list of documents.
    """
    if not docs:
        logger.info("No documents to print.")
        return
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
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
