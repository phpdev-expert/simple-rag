"""Tests for the LlamaIndex retrieval + LangGraph workflow layers."""
import os

from cv_screener.rag import CV_DIR, RetrievedChunk, index_size, retrieve
from cv_screener.rag_graph import RAG_GRAPH, extractive_answer


def test_index_has_all_cvs():
    """The index should contain at least one node per CV (10 CVs)."""
    n_pdfs = len([f for f in os.listdir(CV_DIR) if f.endswith(".pdf")])
    assert n_pdfs == 10
    assert index_size() >= n_pdfs


def test_retrieve_returns_typed_chunks():
    chunks = retrieve("Python backend engineer", k=3)
    assert chunks and all(isinstance(c, RetrievedChunk) for c in chunks)
    assert all(c.candidate and c.text for c in chunks)


def test_retrieval_surfaces_relevant_candidate():
    """A cloud/DevOps query should retrieve the DevOps/cloud candidates."""
    names = {c.candidate for c in retrieve("Kubernetes and cloud infrastructure", k=3)}
    assert names & {"Diego Fernandez", "Fatima Al-Sayed"}


def test_semantic_retrieval_finds_ml_candidates():
    """Semantic search finds ML people for a query without literal keywords."""
    names = {c.candidate for c in retrieve("who can build generative AI chatbots", k=3)}
    assert names & {"Marcus Boateng", "Nadia Petrova"}


def test_extractive_answer_node():
    """The pre-LLM node builds a retrieval_answer from retrieved chunks."""
    chunks = retrieve("PyTorch", k=2)
    out = extractive_answer({"question": "PyTorch", "documents": chunks})
    assert out["retrieval_answer"].startswith("Top matches for:")


def test_graph_returns_both_answers_without_llm():
    """Without an LLM key: retrieval_answer set, llm_answer is None."""
    result = RAG_GRAPH.invoke({"question": "Who knows Python?", "k": 3})
    assert result["retrieval_answer"]
    assert result["llm_answer"] is None
    assert len(result["documents"]) > 0
