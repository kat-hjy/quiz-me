"""Anatomy Scenario Prompt Strategy."""

from dotenv import load_dotenv
from loguru import logger
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Union
from langgraph.pregel.io import AddableValuesDict
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
import io
import re
import base64

from IPython.display import HTML, display
from PIL import Image
from langchain.retrievers.multi_vector import MultiVectorRetriever

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
        retriever: MultiVectorRetriever,
        prompt: PromptTemplate,
        llm: Union[ChatAnthropic, any],
    ) -> None:
        """Initialize the Anatomy Scenario Prompt Strategy.

        Args:
            retriever (MultiVectorRetriever): The retriever initialized with the vector store.
            prompt (PromptTemplate): The prompt template to generate the prompt.
            llm (Union[ChatAnthropic, any]): The language model to generate the response.
        """
        self.retriever = retriever
        self.prompt = prompt
        self.llm = llm

    def _plt_img_base64(self, img_base64: str) -> None:
        """Disply base64 encoded string as image

        Args:
            img_base64: Base64 encoded string
        """
        # Create an HTML img tag with the base64 string as the source
        image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'
        # Display the image by rendering the HTML
        display(HTML(image_html))

    def _looks_like_base64(self, sb):
        """Check if the string looks like base64

        Args:
            sb: String to check
        """
        return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None

    def _is_image_data(self, b64data) -> bool:
        """
        Check if the base64 data is an image by looking at the start of the data

        Args:
            b64data: Base64 encoded

        Returns:
            bool: True if the data is an image
        """
        image_signatures = {
            b"\xff\xd8\xff": "jpg",
            b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a": "png",
            b"\x47\x49\x46\x38": "gif",
            b"\x52\x49\x46\x46": "webp",
        }
        try:
            header = base64.b64decode(b64data)[:8]  # Decode and get the first 8 bytes
            for sig, format in image_signatures.items():
                if header.startswith(sig):
                    return True
            return False
        except Exception:
            return False

    def _resize_base64_image(self, base64_string: str, size: tuple = (128, 128)) -> str:
        """
        Resize an image encoded as a Base64 string.

        Args:
            base64_string: Base64 encoded image
            size: Size to resize the image to

        Returns:
            str: Resized image as a Base64 encoded string
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

    def _split_image_text_types(self, docs: list[Document | str]) -> dict:
        """
        Split base64-encoded images and texts

        Args:
            docs: List of documents

        Returns:
            dict: Dictionary containing images and texts
        """
        b64_images = []
        texts = []
        for doc in docs:
            # Check if the document is of type Document and extract page_content if so
            if isinstance(doc, Document):
                doc = doc.page_content
            if self._looks_like_base64(doc) and self._is_image_data(doc):
                doc = self._resize_base64_image(doc, size=(1300, 600))
                b64_images.append(doc)
            else:
                texts.append(doc)
        return {"images": b64_images, "texts": texts}

    def _format_docs_with_id(self, docs: list[str]) -> str:
        """Format the documents with source IDs.

        Args:
            docs (list[str]): List of documents.

        Returns:
            str: Formatted string with source IDs.
        """
        logger.debug(f"Formatting documents with source IDs: {docs}")
        formatted = [f"Source ID: {i}\n{doc}" for i, doc in enumerate(docs)]
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
            logger.debug(f"Retrieving documents for topic: {state['topic']}")
            logger.debug(f"Retriever: {self.retriever}")

            retrieved_docs = self.retriever.invoke(state["topic"], k=10)
            # Decode any bytes content back to strings
            decoded_docs = []
            for doc in retrieved_docs:
                logger.debug(f"Doc before decoding: {doc}")
                if isinstance(doc, Document):
                    # If the page_content is bytes, decode it
                    if isinstance(doc.page_content, bytes):
                        doc.page_content = doc.page_content.decode("utf-8")
                elif isinstance(doc, bytes):
                    # If the doc itself is bytes, decode it
                    doc = doc.decode("utf-8")
                logger.debug(f"Doc after decoding: {doc}")
                decoded_docs.append(doc)

            # Store the original documents for citation purposes
            return {"context": decoded_docs}

        def process_context(state: State) -> dict:
            """Process the context to separate images and text.

            Args:
                state (State): The state of the conversation.

            Returns:
                dict: The processed context with images and text.
            """
            # Process to separate images and text
            processed_context = self._split_image_text_types(state["context"])
            return {"processed_context": processed_context}

        def format_messages(state: State) -> dict:
            """Format messages with context and images.

            Args:
                state (State): The state of the conversation.

            Returns:
                list: List of HumanMessage objects.
            """
            # Get the formatted context with source IDs
            formatted_context = self._format_docs_with_id(
                state.get("processed_context", {}).get("texts", [])
            )
            logger.debug(f"Formatted context: {formatted_context}")
            messages = []

            # Adding image(s) to the messages if present
            if state.get("processed_context", {}).get("images", {}):
                logger.debug("Adding image(s) to the messages")
                logger.debug(f"state.keys():\n {state.keys()}")
                logger.debug(
                    f"type of state[processed_context]: {type(state['processed_context'])}"
                )
                logger.debug(
                    f"state[processed_context].keys(): {state['processed_context'].keys()}"
                )

                for image in state["processed_context"]["images"]:
                    image_message = {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                    }
                    messages.append(image_message)

            # Format the template with variables
            text_message = {
                "type": "text",
                "text": self.prompt.format(
                    topic=state["topic"], texts=formatted_context
                ),
            }
            messages.append(text_message)
            return {"messages": [HumanMessage(content=messages)]}

        def generate(state: State) -> dict:
            """Generate the response to the user query.

            Args:
                state (State): The state of the conversation.

            Returns:
                dict: The response to the user query.
            """
            # Format messages with context and images
            messages = state["messages"]

            # Just get the raw response from the LLM
            response = self.llm.invoke(messages)
            logger.debug(f"MCQ response: {response}")

            # Use structured output to get citations
            # structured_llm: Runnable = self.llm.with_structured_output(QuotedAnswer)
            # response = structured_llm.invoke(messages)
            # logger.debug(f"Prompt response: {response}")

            return {
                "answer": response
            }  # TODO: check if answer: response.answer or response

        # Create graph using sequence and explicit edges
        graph_builder = StateGraph(State)

        # Add nodes
        graph_builder.add_node("retrieve", retrieve)
        graph_builder.add_node("process_context", process_context)
        graph_builder.add_node("format_messages", format_messages)
        graph_builder.add_node("generate", generate)

        # Connect nodes
        graph_builder.add_edge(START, "retrieve")
        graph_builder.add_edge("retrieve", "process_context")
        graph_builder.add_edge("process_context", "format_messages")
        graph_builder.add_edge("format_messages", "generate")
        graph_builder.set_finish_point("generate")

        self.graph = graph_builder.compile()
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
