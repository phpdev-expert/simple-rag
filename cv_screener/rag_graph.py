"""RAG Q&A workflow — coordinated by LangGraph.

    START -> retrieve -> extractive_answer -> llm_answer -> END

Layer responsibilities:
  * LlamaIndex (cv_screener.rag)  -> retrieve the right CV chunks
  * LangChain (prompt + model)    -> generate the grounded answer
  * LangGraph (this module)       -> wire the steps into a workflow

Two answers are produced from the same retrieved context:
  * retrieval_answer -- BEFORE the LLM call: extractive, straight from retrieval.
  * llm_answer       -- AFTER the LLM call: grounded generation, or None if no
    LLM key is configured.
"""
from __future__ import annotations

from typing import List, Optional, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

from .llm import get_llm
from .rag import RetrievedChunk, retrieve as llamaindex_retrieve


class RAGState(TypedDict, total=False):
    question: str
    k: int
    documents: List[RetrievedChunk]
    retrieval_answer: str      # before LLM
    llm_answer: Optional[str]  # after LLM


def _format_context(chunks: List[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{c.candidate}] {c.text}" for c in chunks)


def retrieve(state: RAGState) -> RAGState:
    """LlamaIndex retrieval step."""
    chunks = llamaindex_retrieve(state["question"], k=state.get("k", 4))
    return {"documents": chunks}


def extractive_answer(state: RAGState) -> RAGState:
    """Pre-LLM answer: ranked candidates + top snippet each (no LLM)."""
    chunks = state["documents"]
    if not chunks:
        return {"retrieval_answer": "No matching CVs found."}

    seen: dict[str, str] = {}
    for c in chunks:
        if c.candidate not in seen:
            seen[c.candidate] = c.text.replace("\n", " ")[:200]

    lines = [f"Top matches for: {state['question']}", ""]
    for i, (name, snippet) in enumerate(seen.items(), start=1):
        lines.append(f"{i}. {name} — {snippet}…")
    return {"retrieval_answer": "\n".join(lines)}


_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You answer questions about job candidates using ONLY the provided CV "
     "excerpts. Cite candidates by name. If the excerpts don't contain the "
     "answer, say so plainly."),
    ("human", "CV excerpts:\n{context}\n\nQuestion: {question}"),
])


def llm_answer(state: RAGState) -> RAGState:
    """Post-LLM answer via LangChain, or None when no LLM is configured."""
    llm = get_llm()
    if llm is None:
        return {"llm_answer": None}
    context = _format_context(state["documents"])
    chain = _ANSWER_PROMPT | llm
    try:
        result = chain.invoke({"context": context, "question": state["question"]})
        return {"llm_answer": result.content}
    except Exception as exc:  # noqa: BLE001 - surface LLM/auth errors, don't crash the API
        return {"llm_answer": f"[LLM call failed: {exc}]"}


def build_rag_graph():
    builder = StateGraph(RAGState)
    builder.add_node("retrieve", retrieve)
    builder.add_node("extractive_answer", extractive_answer)
    builder.add_node("llm_answer", llm_answer)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "extractive_answer")
    builder.add_edge("extractive_answer", "llm_answer")
    builder.add_edge("llm_answer", END)
    return builder.compile()


RAG_GRAPH = build_rag_graph()
