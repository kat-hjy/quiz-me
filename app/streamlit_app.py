"""MCQ Generator App"""

import streamlit as st
from quiz_me.pipelines import general_utils as gu
from quiz_me.pipelines.main import Pipeline
from quiz_me.prompts import utils as pu
from dotenv import load_dotenv
from loguru import logger
from io import BytesIO
import base64

load_dotenv()

st.title("🦜🔗 MCQ Generator App")

anthropic_api_key = st.sidebar.text_input("Anthropic API Key", type="password")
config = gu.load_config()
model_category = config["model"]["category"]


def generate_response(input_text, prompt_strategy):
    pipeline = Pipeline()
    success, output = pipeline.main(topic=input_text, prompt_strategy=prompt_strategy)
    if success:  # if the response was generated successfully in json format
        logger.info("Response generated successfully!")
    else:
        logger.error("Failed to generate response.")
    dict_output = dict(output)
    logger.info(f"Input: {input_text}")
    logger.info(f"Output: {dict_output}")
    logger.info(f"Content: {dict_output["answer"]}")
    try:
        answer: dict = dict(dict_output["answer"])
        logger.info(f"Answer: {answer["content"]}")
    except ValueError:
        answer = dict_output["answer"]
        logger.info(f"Answer: {answer}")

    if not output:
        output = "No response generated."
    content = answer["content"] if isinstance(answer, dict) else answer
    formatted_content = content.replace("\n", "  \n")
    st.subheader("Generated MCQs:")
    st.info(formatted_content)

    # Display the context
    st.subheader("Sources:")
    context_list: list[str] = list(dict_output["context"])
    for i, c in enumerate(context_list):
        if pu._is_base64(c):
            logger.info("Base64 image detected.")
            bytes = BytesIO(base64.b64decode(c))
            if "DeltaGenerator" in bytes:
                logger.info("DeltaGenerator detected in base64 image.")
                logger.info(f"Old c: {len(c)}")
                # remove the DeltaGenerator portion
                # e.g. DeltaGenerator(_provided_cursor=LockedCursor(_index=7, _parent_path=(1,), _props={'delta_type': 'imgs', 'add_rows_metadata': None}), _parent=DeltaGenerator(_provided_cursor=RunningCursor(_parent_path=(1,), _index=8), _parent=DeltaGenerator(), _block_type='form', _form_data=FormData(form_id='my_form')))
                c = c.split("DeltaGenerator")[0]
                logger.info(f"New c: {len(c)}")
            with st.expander(label=f"Source {i}", expanded=False):
                st.image(bytes)
        elif "DeltaGenerator" in c:
            logger.info("DeltaGenerator detected.")
            logger.info(c)
            pass
        else:
            logger.info("Text detected.")
            with st.expander(label=f"Source {i}", expanded=False):
                st.info(c)


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
    logger.info(f"mcq_type: {mcq_type}")
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
