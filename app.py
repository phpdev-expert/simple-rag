"""FastAPI service exposing the CV RAG Q&A bot.

Run:
    python app.py                 # uses PORT env (default 8000)
    PORT=8100 python app.py       # pick another port if 8000 is taken
    uvicorn app:app --port 8100   # or drive uvicorn directly
    # then open http://127.0.0.1:<port>/docs

POST /ask returns TWO answers for the same question:
  * answer_before_llm -- extractive, straight from Chroma retrieval (no LLM)
  * answer_after_llm  -- grounded generation from the configured LLM (or null)
"""
from __future__ import annotations

from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field

load_dotenv()  # must run before the LLM factory reads the environment

from cv_screener.llm import active_model_label, has_api_key  # noqa: E402
from cv_screener.rag_graph import RAG_GRAPH  # noqa: E402

app = FastAPI(
    title="CV RAG Q&A API",
    description="Ask questions about the candidate CVs. Returns a pre-LLM "
    "(retrieval) answer and a post-LLM (generated) answer.",
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str = Field(..., examples=["Who has Kubernetes and cloud experience?"])
    k: int = Field(4, ge=1, le=10, description="Number of chunks to retrieve")


class AskResponse(BaseModel):
    question: str
    retrieved_from: List[str] = Field(description="Candidates the context came from")
    answer_before_llm: str = Field(description="Extractive answer from retrieval only")
    answer_after_llm: Optional[str] = Field(
        description="LLM-generated answer, or null if no LLM key is configured"
    )
    llm_used: Optional[str] = Field(description="Active model id, or null in fallback mode")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "llm_configured": has_api_key(), "model": active_model_label()}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    result = RAG_GRAPH.invoke({"question": req.question, "k": req.k})
    sources = sorted({c.candidate for c in result["documents"]})
    return AskResponse(
        question=req.question,
        retrieved_from=sources,
        answer_before_llm=result["retrieval_answer"],
        answer_after_llm=result.get("llm_answer"),
        llm_used=active_model_label(),
    )


if __name__ == "__main__":
    import os

    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="127.0.0.1", port=port)
