"""
tools/llm.py
Single Groq LLM instance shared by all agents.
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from tools.search import parse_json

load_dotenv()

llm = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key="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",
    model="llama-3.3-70b-versatile",
    temperature=0,           # deterministic → more reliable JSON
)


def call_llm_json(system_prompt: str, user_message: str, fallback):
    """Call the LLM and parse JSON from the response."""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]
    response = llm.invoke(messages)
    return parse_json(response.content, fallback)
