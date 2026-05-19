"""
tools/llm.py
Single Groq LLM instance shared by all agents.
"""
import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from tools.search import parse_json

load_dotenv()

# ── reads from Streamlit secrets (deployed) or .env (local) ──
_api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

llm = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=_api_key,                  # ← no hardcoded key
    model="llama-3.3-70b-versatile",
    temperature=0,
)

def call_llm_json(system_prompt: str, user_message: str, fallback):
    """Call the LLM and parse JSON from the response."""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]
    response = llm.invoke(messages)
    return parse_json(response.content, fallback)
