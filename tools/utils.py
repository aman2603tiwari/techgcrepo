"""
tools/utils.py
Utility helpers for the Conference Agent.
"""

import os
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError


def get_secret(key: str, env_key: str | None = None) -> str | None:
    """Return a secret from Streamlit or fall back to an environment variable."""
    env_key = env_key or key
    try:
        return st.secrets[key]
    except (StreamlitSecretNotFoundError, KeyError):
        return os.environ.get(env_key)
