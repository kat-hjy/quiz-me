"""This module contains the main pipeline for the project."""

from langchain_core.documents import Document
from langchain_core.vectorstores.base import VectorStoreRetriever
from dotenv import load_dotenv
from loguru import logger
from langchain_chroma import Chroma
from langchain_anthropic import ChatAnthropic
from langchain.retrievers.multi_vector import MultiVectorRetriever
from pathlib import Path

# TODO: import directly from modules
from quiz_me.pipelines.indexing import TextIndexing, MultiModalIndexing  # noqa: E402
from quiz_me.prompts.prompt_strategy_pattern import PromptContext  # noqa: E402
from quiz_me.prompts.prompt_strategies.flashcards import Flashcards  # noqa: E402
from quiz_me.prompts.prompt_strategies.microbiology_scenario import (  # noqa: E402
    MicrobiologyScenario,
)
from quiz_me.prompts.prompt_strategies.anatomy_scenario import AnatomyScenario  # noqa: E402
from quiz_me.modeling import utils as mu  # noqa: E402
from quiz_me.prompts import utils as pu  # noqa: E402
import quiz_me.pipelines.general_utils as gu  # noqa: E402

# Load environment variables
load_dotenv()


class Pipeline:
    """Main pipeline for the project."""

    def __init__(self):
        """Initialize the Pipeline class."""
        pass

    def main(
        self, topic: str, prompt_strategy: str = "anatomy_scenario"
    ) -> tuple[bool, any]:
        """Main pipeline for the project.

        Args:
            topic (str): The topic to generate a response for.
            prompt_strategy (str, optional): The prompt strategy to use. Defaults to "anatomy_scenario".

        Raises:
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_

        Returns:
            str: _description_
        """
        # Load configuration and catalog
        catalog: dict = gu.load_catalog()
        config: dict = gu.load_config()

        # Instantiate TextIndexing object
        idx: TextIndexing = TextIndexing(
            vectorstore_dir=catalog["retrieval"]["vectorstore"]["persist_directory"],
            collection_name=config["indexing"]["text"]["collection_name"],
        )
        # Instantiate LLM object
        model_category: str = config["model"]["category"]
        if model_category == "anthropic":
            llm: ChatAnthropic = mu.get_anthropic_llm(
                model_name=config["model"]["anthropic"]["model_name"],
                temperature=config["model"]["anthropic"]["temperature"],
                timeout=config["model"]["anthropic"]["timeout"],
                max_retries=config["model"]["anthropic"]["max_retries"],
                max_tokens=config["model"]["anthropic"]["max_tokens"],
            )
        else:
            raise ValueError(f"Model category {model_category} not recognized.")
        mmi: MultiModalIndexing = MultiModalIndexing(
            docstore_dir=catalog["retrieval"]["docstore"]["root_path"],
            vectorstore_dir=catalog["retrieval"]["vectorstore"]["persist_directory"],
            collection_name=config["indexing"]["multimodal"]["collection_name"],
            llm=llm,
        )

        ### INDEXING ###
        logger.info("Indexing...")
        if config["indexing"]["type"] == "text":
            if config["indexing"]["activate"]:
                logger.info("Indexing text only...")
                docs: list[Document] = idx.load_documents(
                    catalog["indexing"]["text"]["learning_material_dir"]
                )
                docs: list[Document] = idx.preprocess_documents(docs)
                docs: list[Document] = idx.split_documents(docs)
                vectorstore: Chroma = idx.store_documents(
                    vectorstore_dir=catalog["retrieval"]["vectorstore"][
                        "persist_directory"
                    ],
                    collection_name=config["indexing"]["text"]["collection_name"],
                    docs=docs,
                )
            else:
                logger.info(
                    "Indexing is not activated. Getting vectorstore from memory..."
                )
                vectorstore: Chroma = idx.get_vectorstore()
        elif config["indexing"]["type"] == "multimodal":
            if config["indexing"]["activate"]:
                pdf_dir_path = Path(
                    catalog["indexing"]["multimodal"]["pdf_dir_path"]
                ).resolve()
                figures_path = Path(
                    catalog["indexing"]["multimodal"]["figures_path"]
                ).resolve()
                logger.info("Indexing multimodal embeddings...")
                logger.debug(f"PDF directory path: {str(pdf_dir_path)}")
                logger.debug(f"Figures path: {str(figures_path)}")
                mmi.load_documents(str(pdf_dir_path), str(figures_path))
                vectorstore: Chroma = mmi.get_vectorstore()
            else:
                logger.info(
                    "Indexing is not activated. Getting vectorstore from memory..."
                )
                vectorstore: Chroma = mmi.get_vectorstore()
        else:
            raise ValueError(
                f"Indexing type {config['indexing']['type']} not recognized."
            )

        if config["generation"]["activate"]:
            ### RETRIEVAL ###
            # TODO: use Strategy Pattern for indexing
            if config["indexing"]["type"] == "text":
                retriever: VectorStoreRetriever = idx.get_retriever(vectorstore)
            elif config["indexing"]["type"] == "multimodal":
                retriever: MultiVectorRetriever = mmi.get_retriever(vectorstore)

            ### GENERATION ###
            if not prompt_strategy:
                prompt_strategy: str = config["generation"]["prompt_strategy"]
            with open(catalog["generation"][prompt_strategy]["template"], "r") as file:
                prompt_content = file.read()
                logger.debug(f"Prompt content: \n{prompt_content}")
            if prompt_strategy == "flashcards":
                prompt = pu.get_prompt(prompt_content)
                prompt_context: PromptContext = PromptContext(
                    strategy=Flashcards(retriever=retriever, llm=llm, prompt=prompt)
                )
                output = prompt_context.generate_response(
                    {"topic": config["generation"][prompt_strategy]["topic"]}
                )
            elif prompt_strategy == "microbiology_scenario":
                prompt = pu.get_prompt(prompt_content)
                prompt_context: PromptContext = PromptContext(
                    strategy=MicrobiologyScenario(
                        retriever=retriever, llm=llm, prompt=prompt
                    )
                )
            elif prompt_strategy == "anatomy_scenario":
                prompt = pu.get_prompt(template=prompt_content)
                logger.info("Generating response using AnatomyScenario...")
                prompt_context: PromptContext = PromptContext(
                    strategy=AnatomyScenario(
                        retriever=retriever, llm=llm, prompt=prompt
                    )
                )
                if not topic:
                    raise ValueError("Topic not provided.")
                output = prompt_context.generate_response({"topic": topic})
            else:
                raise ValueError(f"Prompt strategy {prompt_strategy} not recognized.")

            ### POST-GENERATION ###
            logger.info(f"Output: \n{output}")
            # save output to a json file
            success, output = gu.save_output(
                output, catalog["generation"][prompt_strategy]["output_dir"]
            )
            logger.info(f"Success: {success}")
            logger.info(f"Output: {output}")

            return success, output
        else:
            logger.info("Generation is not activated. Skipping...")
            success = False
            output = None
            return success, output


if __name__ == "__main__":
    # Load configuration and catalog
    catalog: dict = gu.load_catalog()
    config: dict = gu.load_config()
    prompt_strategy = config["generation"]["prompt_strategy"]
    topic = config["generation"][prompt_strategy]["topic"]
    pipeline = Pipeline()
    pipeline.main(topic=topic, prompt_strategy=prompt_strategy)
