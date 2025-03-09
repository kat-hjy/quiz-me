import streamlit as st
from langchain_anthropic import ChatAnthropic
import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

st.title("🦜🔗 Quickstart App")

openai_api_key = st.sidebar.text_input("Anthropic API Key", type="password")


def generate_response(input_text):
    model = ChatAnthropic()
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
