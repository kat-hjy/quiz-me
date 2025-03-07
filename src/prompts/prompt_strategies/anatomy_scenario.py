"""Anatomy Scenario Prompt Strategy."""

from dotenv import load_dotenv
from loguru import logger
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Union
from langgraph.pregel.io import AddableValuesDict
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langchain_core.runnables import Runnable
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_core.messages import HumanMessage
from src.prompts import utils

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


class State(TypedDict, total=False):
    """State of the conversation."""

    topic: str
    context: list[Document]
    processed_context: list
    messages: list
    answer: QuotedAnswer


class AnatomyScenario:
    def __init__(
        self,
        retriever: VectorStoreRetriever,
        prompt: PromptTemplate,
        llm: Union[ChatAnthropic, any],
    ) -> None:
        """Initialize the Anatomy Scenario Prompt Strategy.

        Args:
            retriever (VectorStoreRetriever): The retriever initialized with the vector store.
            prompt (PromptTemplate): The prompt template to generate the prompt.
            llm (Union[ChatAnthropic, any]): The language model to generate the response.
        """
        self.retriever = retriever
        self.prompt = prompt
        self.llm = llm

    def _format_docs_with_id(docs: list[Document]) -> str:
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
            # Get raw documents from retriever
            retrieved_docs = self.retriever.invoke(state["topic"])
            # Store the original documents for citation purposes
            return {"context": retrieved_docs}

        def process_context(state: State) -> dict:
            """Process the context to separate images and text.

            Args:
                state (State): The state of the conversation.

            Returns:
                dict: The processed context with images and text.
            """
            # Process to separate images and text
            processed_context = utils.split_image_text_types(state["context"])
            return {"processed_context": processed_context}

        def format_messages(state: State) -> list:
            """Format messages with context and images.

            Args:
                state (State): The state of the conversation.

            Returns:
                list: List of HumanMessage objects.
            """
            # Get the formatted context with source IDs
            formatted_context = self._format_docs_with_id(
                state.get("processed_context", {}).get("text", [])
            )

            # Extract image data from the processed context if available
            images = state.get("processed_context", {}).get("images", [])
            logger.debug(f"Formatted context: \n{formatted_context}")
            logger.debug(f"Images: \n{images}")

            content = []

            # Generate the prompt text using the template
            prompt_text = self.prompt.invoke({"context": formatted_context})

            content = []

            # Adding image(s) to the content if present
            if images and len(images) > 0:
                image_content = {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": images[0],
                    },
                }
                content.append(image_content)

            # Adding the text part generated from the prompt template
            text_content = {"type": "text", "text": prompt_text}
            content.append(text_content)

            return [HumanMessage(content=content)]

        def generate(state: State) -> dict:
            """Generate the response to the user query.

            Args:
                state (State): The state of the conversation.

            Returns:
                dict: The response to the user query.
            """
            # Format messages with context and images
            messages: list = format_messages(state)

            # Use structured output to get citations
            structured_llm: Runnable = self.llm.with_structured_output(QuotedAnswer)
            response = structured_llm.invoke(messages)
            logger.debug(f"Prompt response: {response}")

            return {
                "answer": response
            }  # TODO: check if answer: response.answer or response

        # Create graph using sequence and explicit edges
        graph_builder: StateGraph = StateGraph(State)
        graph_builder.add_node("retrieve", retrieve)
        graph_builder.add_node("process_context", process_context)
        graph_builder.add_node("generate", generate)

        # Connect the nodes
        graph_builder.add_edge(START, "retrieve")
        graph_builder.add_edge("retrieve", "process_context")
        graph_builder.add_edge("process_context", "generate")
        graph_builder.set_finish_point("generate")

        self.graph: CompiledStateGraph = graph_builder.compile()
        return self.graph

    def generate_output(
        self, input: dict, graph: CompiledStateGraph = None
    ) -> AddableValuesDict:
        """Process a user query and generate the output through the RAG pipeline.

        Args:
            input (dict): topic input

        Returns:
            AddableValuesDict: Dictionary containing the final output with citations
        """
        topic: str = input.get("topic", "")

        if self.graph is None:
            self.graph = graph

        state: dict = {"topic": topic}
        result: AddableValuesDict = self.graph.invoke(state)
        return result
