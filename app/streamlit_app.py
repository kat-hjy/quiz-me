import streamlit as st
from langchain_anthropic import ChatAnthropic
from quiz_me.modeling import utils as mu
from quiz_me.pipelines import general_utils as gu
from dotenv import load_dotenv

load_dotenv()

st.title("🦜🔗 Quickstart App")

openai_api_key = st.sidebar.text_input("Anthropic API Key", type="password")

config = gu.load_config()
model_category = config["model"]["category"]


def generate_response(input_text):
    if model_category == "anthropic":
        model: ChatAnthropic = mu.get_anthropic_llm(
            model_name=config["model"]["anthropic"]["model_name"],
            temperature=config["model"]["anthropic"]["temperature"],
            timeout=config["model"]["anthropic"]["timeout"],
            max_retries=config["model"]["anthropic"]["max_retries"],
            max_tokens=config["model"]["anthropic"]["max_tokens"],
        )
    st.info(model.invoke(input_text))


with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
        "What are the three key pieces of advice for learning how to code?",
    )
    submitted = st.form_submit_button("Submit")
    if not openai_api_key.startswith("sk-ant-api"):
        st.warning("Please enter your Anthropic API key!", icon="⚠")
    if submitted and openai_api_key.startswith("sk-ant-api"):
        generate_response(text)
