"""This module contains utility functions for the prompts."""

from langchain_core.prompts import PromptTemplate

import base64
import io
from PIL import Image
# import numpy as np
# from io import BytesIO


def get_prompt(template: str) -> PromptTemplate:
    """Get a PromptTemplate object from a template string.

    Args:
        template (str): Template string.

    Returns:
        PromptTemplate: PromptTemplate object.
    """
    prompt = PromptTemplate.from_template(template)
    return prompt


def _resize_base64_image(base64_string: str, size: tuple = (128, 128)) -> str:
    """
    Resize an image encoded as a Base64 string.

    Args:
        base64_string (str): Base64 string of the original image.
        size (tuple): Desired size of the image as (width, height).

    Returns:
        str: Base64 string of the resized image.
    """
    # Decode the Base64 string
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))

    # Resize the image
    resized_img = img.resize(size, Image.LANCZOS)

    # Save the resized image to a bytes buffer
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)

    # Encode the resized image to Base64
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _is_base64(s: str) -> bool:
    """Check if a string is Base64 encoded

    Args:
        s (str): String to check

    Returns:
        bool: True if the string is Base64 encoded, False otherwise
    """
    try:
        return base64.b64encode(base64.b64decode(s)) == s.encode()
    except Exception:
        return False


def split_image_text_types(docs: list) -> dict[str, list]:
    """Split numpy array images and texts

    Args:
        docs (list): List of Document objects

    Returns:
        dict[str, list]: Dictionary of images and texts
    """
    images = []
    text = []
    for doc in docs:
        doc = doc.page_content  # Extract Document contents
        if _is_base64(doc):
            # Resize image to avoid OAI server error
            images.append(
                _resize_base64_image(doc, size=(250, 250))
            )  # base64 encoded str
        else:
            text.append(doc)
    return {"images": images, "texts": text}
