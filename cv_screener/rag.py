"""Retrieval layer — powered by LlamaIndex.

LlamaIndex owns the "find the right information" job end to end:
  * SimpleDirectoryReader  -> read the CV PDFs (candidate tagged per file)
  * SentenceSplitter       -> chunk them into nodes
  * HuggingFaceEmbedding   -> embed nodes locally (offline, no API key)
  * ChromaVectorStore      -> persist vectors in Chroma at data/chroma/
  * VectorStoreIndex       -> retrieve the top-k nodes for a query

LangChain (prompts + models) and LangGraph (workflow) live in the other modules
and consume the `RetrievedChunk` list this module returns.
"""
from __future__ import annotations

import glob
import os
import re
import shutil
from dataclasses import dataclass
from functools import lru_cache

_ROOT = os.path.dirname(os.path.dirname(__file__))
CV_DIR = os.path.join(_ROOT, "data", "cvs")
PERSIST_DIR = os.path.join(_ROOT, "data", "chroma")
COLLECTION = "cvs"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class RetrievedChunk:
    """A retrieval hit, decoupled from any framework's document type."""

    candidate: str
    source: str
    text: str
    score: float


def _candidate_name(path: str) -> str:
    """'.../01_ava_chen.pdf' -> 'Ava Chen'."""
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = re.sub(r"^\d+_", "", stem)
    return stem.replace("_", " ").title()


def _file_metadata(path: str) -> dict:
    return {"candidate": _candidate_name(path), "source": os.path.basename(path)}


def _configure_settings() -> None:
    """Point LlamaIndex at the local embedding model; we don't use its LLM."""
    from llama_index.core import Settings
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    Settings.llm = None  # generation is handled by LangChain, not LlamaIndex


@lru_cache(maxsize=1)
def _chroma_collection():
    import chromadb

    client = chromadb.PersistentClient(path=PERSIST_DIR)
    return client.get_or_create_collection(COLLECTION)


@lru_cache(maxsize=1)
def get_index():
    """Open the LlamaIndex vector index, populating Chroma on first use."""
    from llama_index.core import (
        SimpleDirectoryReader,
        StorageContext,
        VectorStoreIndex,
    )
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.vector_stores.chroma import ChromaVectorStore

    _configure_settings()
    collection = _chroma_collection()
    vector_store = ChromaVectorStore(chroma_collection=collection)

    if collection.count() == 0:
        if not glob.glob(os.path.join(CV_DIR, "*.pdf")):
            raise FileNotFoundError(
                f"No CVs in {CV_DIR}. Run `python generate_cvs.py` first."
            )
        documents = SimpleDirectoryReader(
            input_dir=CV_DIR, file_metadata=_file_metadata
        ).load_data()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            transformations=[SentenceSplitter(chunk_size=512, chunk_overlap=64)],
        )

    return VectorStoreIndex.from_vector_store(vector_store)


def retrieve(question: str, k: int = 4) -> list[RetrievedChunk]:
    """Return the top-k most relevant CV chunks for the question."""
    nodes = get_index().as_retriever(similarity_top_k=k).retrieve(question)
    return [
        RetrievedChunk(
            candidate=n.node.metadata.get("candidate", "?"),
            source=n.node.metadata.get("source", "?"),
            text=n.node.get_content().strip(),
            score=float(n.score or 0.0),
        )
        for n in nodes
    ]


def index_size() -> int:
    return _chroma_collection().count()


def build_index(rebuild: bool = False) -> int:
    """(Re)build the Chroma-backed LlamaIndex. Returns the number of nodes."""
    if rebuild and os.path.isdir(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
        _chroma_collection.cache_clear()
    get_index.cache_clear()
    get_index()  # triggers ingestion if the collection is empty
    return index_size()
