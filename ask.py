"""RAG Q&A bot over the CV corpus.

Usage:
    python ask.py                      # interactive REPL
    python ask.py "who knows Kubernetes?"   # single question
"""
from __future__ import annotations

import sys

from dotenv import load_dotenv

from cv_screener.llm import active_model_label, has_api_key
from cv_screener.rag_graph import RAG_GRAPH

load_dotenv()


def answer(question: str) -> None:
    result = RAG_GRAPH.invoke({"question": question})
    print("\n--- before LLM (retrieval) ---")
    print(result["retrieval_answer"])
    print("\n--- after LLM (generated) ---")
    print(result.get("llm_answer") or "[no LLM key configured — see .env.example]")
    sources = sorted({c.candidate for c in result["documents"]})
    print(f"\n— retrieved from: {', '.join(sources)}\n")


def main() -> None:
    mode = active_model_label() if has_api_key() else "extractive fallback (no LLM key)"
    print(f"CV RAG Q&A — using {mode}.")

    if len(sys.argv) > 1:
        answer(" ".join(sys.argv[1:]))
        return

    print("Ask questions about the candidates. Ctrl-C or empty line to quit.\n")
    try:
        while True:
            question = input("? ").strip()
            if not question:
                break
            answer(question)
    except (EOFError, KeyboardInterrupt):
        print()


if __name__ == "__main__":
    main()
