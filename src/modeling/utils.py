"""This module contains model-related utility functions."""

from langchain_anthropic import ChatAnthropic
from loguru import logger


def get_anthropic_llm(
    model_name: str,
    temperature: float,
    timeout: int | None,
    max_retries: int = 1,
    max_tokens: int = 1024,
):
    """Get an instance of the ChatAnthropic class.

    Args:
        model_name (str): name of the Anthropic model to load
        temperature (float): Temperature to use to generate responses
        timeout (int | None): Time in seconds to wait for a response before timing out.
        max_retries (int, optional): Maximum number of retries to attempt. Defaults to 1.
        max_tokens (int, optional): Maximum number of tokens to output. Defaults to 1024.

    Returns:
        ChatAnthropic: Instance of the ChatAnthropic class
    """
    logger.debug(f"Loading Anthropic model {model_name}")
    logger.debug(f"Temperature: {temperature}")
    logger.debug(f"Timeout: {timeout}")
    logger.debug(f"Max retries: {max_retries}")
    logger.debug(f"Max tokens: {max_tokens}")

    llm = ChatAnthropic(
        model=model_name,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
        max_tokens=max_tokens,
    )
    return llm
