"""Shared test fixtures.

Ensures a built LlamaIndex/Chroma index exists and that tests run WITHOUT any
LLM key so the "after LLM" answer is deterministically null. To test the real
LLM path, set an API key and run the app manually.
"""
import pytest

from cv_screener import llm as llm_mod
from cv_screener.rag import build_index, index_size


@pytest.fixture(autouse=True)
def no_llm_key(monkeypatch):
    """Strip provider keys and reset the cached LLM so no LLM is used."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    llm_mod.get_llm.cache_clear()
    yield
    llm_mod.get_llm.cache_clear()


@pytest.fixture(scope="session", autouse=True)
def ensure_index():
    """Build the vector index once for the whole test session if empty."""
    if index_size() == 0:
        build_index(rebuild=True)
