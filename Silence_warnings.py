"""
silence_warnings.py
Import this at the top of main.py and app.py to suppress
all the transformers/sentence-transformers/HF noise.
"""
import os
import warnings
import logging

# Kill noisy loggers
for name in [
    "sentence_transformers", "transformers", "transformers.modeling_utils",
    "huggingface_hub", "huggingface_hub.file_download",
    "huggingface_hub.utils._headers", "filelock",
]:
    logging.getLogger(name).setLevel(logging.ERROR)

# Kill Python warnings
warnings.filterwarnings("ignore")

# Env vars that suppress HF chatter
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")