"""LLM factory.

Picks a provider from whatever API key is present, in priority order:
  1. OPENROUTER_API_KEY -> OpenRouter (OpenAI-compatible gateway)
  2. GROQ_API_KEY       -> Groq (free tier, OpenAI-compatible)
  3. ANTHROPIC_API_KEY  -> Claude

If none is set, returns None and callers fall back to non-LLM behaviour so the
pipeline still runs offline.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"


def _provider() -> Optional[str]:
    if os.getenv("OPENROUTER_API_KEY"):
        return "openrouter"
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    return None


def has_api_key() -> bool:
    return _provider() is not None


def active_model_label() -> Optional[str]:
    """Human-readable id of the active model, e.g. 'openrouter:openai/gpt-4o'."""
    provider = _provider()
    if provider == "openrouter":
        return f"openrouter:{os.getenv('OPENROUTER_MODEL', DEFAULT_OPENROUTER_MODEL)}"
    if provider == "groq":
        return f"groq:{os.getenv('GROQ_MODEL', DEFAULT_GROQ_MODEL)}"
    if provider == "anthropic":
        return f"anthropic:{DEFAULT_ANTHROPIC_MODEL}"
    return None


@lru_cache(maxsize=1)
def get_llm():
    """Return a chat model instance, or None if no API key is configured."""
    provider = _provider()
    if provider == "openrouter":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL),
            base_url=OPENROUTER_BASE_URL,
            api_key=os.environ["OPENROUTER_API_KEY"],
            temperature=0,
            max_tokens=1024,
            default_headers={
                "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
                "X-Title": os.getenv("OPENROUTER_SITE_NAME", "CV RAG Q&A"),
            },
        )
    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
            temperature=0,
            max_tokens=1024,
        )
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=DEFAULT_ANTHROPIC_MODEL, temperature=0, max_tokens=1024
        )
    return None
