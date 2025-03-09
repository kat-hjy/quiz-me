"""Flashcards prompt strategy."""

from dotenv import load_dotenv
from loguru import logger
from langchain_core.documents import Document
from langchain_core.vectorstores.base import VectorStoreRetriever
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from typing import Union
from typing_extensions import TypedDict
from langgraph.pregel.io import AddableValuesDict
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langchain_core.runnables import (
    Runnable,
    # RunnablePassthrough,
)
from langchain_anthropic import ChatAnthropic

# Load environment variables
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


class Flashcards:
    def __init__(
        self,
        retriever: VectorStoreRetriever,
        prompt: PromptTemplate,
        llm: Union[ChatAnthropic, any],
    ) -> None:
        """Initialize the Flashcards class.

        Args:
            retriever (VectorStoreRetriever): The retriever to use.
            prompt (PromptTemplate): The prompt template to use.
            llm (Union[ChatAnthropic, any]): The language model to use.
        """
        self.retriever = retriever
        self.prompt = prompt
        self.llm = llm

    def format_docs_with_id(self, docs: list[Document]) -> str:
        """Format the documents with source IDs.

        Args:
            docs (list[Document]): List of documents.

        Returns:
            str: Formatted string with source IDs.
        """
        formatted = [
            f"Source ID: {i}\nArticle Title: {doc.metadata.get('title', 'N/A')}\nPage Label: {doc.metadata.get('page_label', 'N/A')}\nPage Content: {doc.page_content}"
            for i, doc in enumerate(docs)
        ]
        return "\n\n" + "\n\n".join(formatted)

    def compile_application(self) -> CompiledStateGraph:
        """Compile the application graph.

        Returns:
            CompiledStateGraph: The compiled application state graph.
        """

        # Define application steps
        def retrieve(state: State) -> dict:
            """Retrieve documents from the retriever.

            Args:
                state (State): The state of the conversation.

            Returns:
                dict: The retrieved documents.
            """
            retrieved_docs = self.retriever.invoke(state["topic"])
            return {"context": retrieved_docs}

        def generate(state: State) -> dict:
            """Generate a response to the user question.

            Args:
                state (State): The state of the conversation.

            Returns:
                dict: The generated response.
            """
            formatted_docs = self.format_docs_with_id(state["context"])
            messages = self.prompt.invoke({"context": formatted_docs})
            structured_llm: Runnable = self.llm.with_structured_output(QuotedAnswer)
            response = structured_llm.invoke(messages)
            logger.debug(f"Prompt response: {response}")
            return {"answer": response.answer}

        graph_builder: StateGraph = StateGraph(State).add_sequence([retrieve, generate])
        graph_builder.add_edge(START, "retrieve")
        self.graph: CompiledStateGraph = graph_builder.compile()
        return self.graph

    def generate_output(
        self, input: dict, graph: CompiledStateGraph = None
    ) -> AddableValuesDict:
        """Generate the output based on the input.

        Args:
            input (dict): The input to generate the output.
            graph (CompiledStateGraph, optional): The graph to use. Defaults to None.

        Returns:
            AddableValuesDict: The generated output.
        """
        logger.debug(f"kwargs: {input}")
        topic: str = input["topic"]

        if graph is None:
            graph = self.graph
        state: dict = {"topic": topic}
        output: AddableValuesDict = graph.invoke(state)
        logger.debug(f"Type of output: {type(output)}")
        return output

    # def get_small_documents(
    #     self, docs: list[Document], char_threshold: int = 10
    # ) -> list[Document]:
    #     """Get all the documents with length of page_content (str) < threshold.

    #     Args:
    #         docs (list[Document]): list of documents.
    #         char_threshold (int, optional): Threshold to consider a document "small". Defaults to 10.

    #     Returns:
    #         list[Document]: list of small documents (page_content < 10 characters).
    #     """
    #     docs = [doc for doc in docs if len(doc.page_content) < char_threshold]
    #     return docs

    # def remove_small_documents(
    #     self, docs: list[Document], char_threshold: int = 10
    # ) -> list[Document]:
    #     """Remove all the documents with length of page_content (str) < threshold.

    #     Args:
    #         docs (list[Document]): list of documents.
    #         char_threshold (int, optional): Threshold to consider a document "small". Defaults to 10.

    #     Returns:
    #         list[Document]: list of documents without small documents (page_content >= 10 characters).
    #     """
    #     docs = [doc for doc in docs if len(doc.page_content) >= char_threshold]
    #     return docs
