"""Microbiology scenario prompt strategy"""

from dotenv import load_dotenv

# from loguru import logger
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores.base import VectorStoreRetriever
from typing import Union

load_dotenv()


class Citation(BaseModel):
    """Citation from a source."""

    source_id: int = Field(
        ...,
        description="The ID of the SPECIFIC source which justifies the answer.",
    )
    title: str = Field(
        ...,
        description="The title of the SPECIFIC source which justifies the answer.",
    )
    quote: str = Field(
        ...,
        description="The VERBATIM quote from the specified source that justifies the answer.",
    )


class QuotedAnswer(BaseModel):
    """Answer the user question based only on the given sources, and cite the sources used."""

    answer: str = Field(
        ...,
        description="The answer to the user question, which is based only on the given sources.",
    )
    citations: list[Citation] = Field(
        ..., description="Citations from the given sources that justify the answer."
    )


class State(TypedDict):
    """State of the conversation."""

    topic: str
    context: list[Document]
    answer: QuotedAnswer


class MicrobiologyScenario:
    """Microbiology scenario prompt strategy."""

    def __init__(
        self,
        retriever: VectorStoreRetriever,
        prompt: PromptTemplate,
        llm: Union[ChatAnthropic, any],
    ) -> None:
        """Initialize the MicrobiologyScenario class.

        Args:
            retriever (VectorStoreRetriever): The retriever to use.
            prompt (PromptTemplate): The prompt template to use.
            llm (Union[ChatAnthropic, any]): The language model to use.
        """
        self.retriever = retriever
        self.prompt = prompt
        self.llm = llm
