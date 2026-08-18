"""
Shared pytest configuration.

This module runs once, before any test file is imported. It stubs the
environment variables the app expects, and replaces the network/model-
loading LLM client classes with lightweight mocks *before* `workflow.py`
is imported anywhere in the test suite.

Why this matters: `workflow.py` instantiates `ChatOpenAI`,
`HuggingFaceEmbeddings`, and `ChatGoogleGenerativeAI` at module level
(not inside a function). Without this patch, simply importing
`workflow` during test collection would try to hit real APIs / download
embedding model weights - slow, flaky, and requires real API keys in CI.
"""
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Make the project root importable (tests/ sits one level below it)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Dummy credentials so any code path that checks "is this set" doesn't
# blow up - no real network calls are made because the classes below
# are mocked before workflow.py binds them.
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "test-key")
os.environ.setdefault("HF_TOKEN", "test-key")
os.environ.setdefault("GOOGLE_API_KEY", "test-key")

import langchain_openai
import langchain_huggingface
import langchain_google_genai

# Replace the classes at their source module BEFORE workflow.py runs
# `from langchain_openai import ChatOpenAI` etc. - Python resolves that
# import by reading the attribute off the already-imported module, so
# patching here is picked up transparently by workflow.py's import.
langchain_openai.ChatOpenAI = MagicMock(name="ChatOpenAI")
langchain_huggingface.HuggingFaceEmbeddings = MagicMock(name="HuggingFaceEmbeddings")
langchain_google_genai.ChatGoogleGenerativeAI = MagicMock(name="ChatGoogleGenerativeAI")
