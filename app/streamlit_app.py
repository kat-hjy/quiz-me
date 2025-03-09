"""Streamlit app"""

import streamlit as st
from quiz_me.pipelines import general_utils as gu
from quiz_me.pipelines.main import Pipeline
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

st.title("🦜🔗 Quickstart App")

anthropic_api_key = st.sidebar.text_input("Anthropic API Key", type="password")
config = gu.load_config()
model_category = config["model"]["category"]


def generate_response(input_text, prompt_strategy):
    pipeline = Pipeline()
    output = pipeline.main(topic=input_text, prompt_strategy=prompt_strategy)
    dict_output = dict(output)
    logger.info(f"Input: {input_text}")
    logger.info(f"Output: {dict_output}")
    logger.info(f"Content: {dict_output["answer"]}")
    try:
        answer: dict = dict(dict_output["answer"])
    except ValueError:
        answer = dict_output["answer"]

    # context_list: list[str] = list(dict_output["context"])
    # for i, c in enumerate(context_list):
    #     if pu._is_base64(c):
    #         bytes = BytesIO(base64.b64decode(c))
    #         st.info(f"Source {i+1}: {st.image(bytes)}")
    #     elif c.contains("DeltaGenerator"):
    #         st.info(f"Source {i+1}: {c}")
    #     else:
    #         # don't display the source
    #         pass
    if not output:
        output = "No response generated."
    st.info(answer["content"] if isinstance(answer, dict) else answer)


with st.form("my_form"):
    text = st.text_area(
        "Enter topic:",
        "Digestive system",
    )
    # add a dropdown of types of questions to generate
    mcq_type = st.selectbox(
        "Select type",
        [
            "Anatomy MCQs",
            # "Microbiology MCQs",
            # "Flashcards",
        ],
    )
    submitted = st.form_submit_button("Submit")
    logger.info(f"Submitted: {submitted}")
    logger.info("mcq_type: ", mcq_type)
    logger.info(f"Text: {text}")
    if not anthropic_api_key.startswith("sk-ant-api"):
        st.warning("Please enter your Anthropic API key!", icon="⚠")
    if submitted and anthropic_api_key.startswith("sk-ant-api"):
        prompt_strategy = ""
        if mcq_type == "Anatomy":
            prompt_strategy = "anatomy_scenario"
        elif mcq_type == "Microbiology":
            prompt_strategy = "microbiology_scenario"
        elif mcq_type == "Flashcards":
            prompt_strategy = "flashcards"
        generate_response(text, prompt_strategy)
