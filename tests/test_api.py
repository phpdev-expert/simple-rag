"""Tests for the FastAPI endpoints."""
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    # no key configured in tests
    assert body["llm_configured"] is False


def test_ask_returns_two_answers():
    resp = client.post(
        "/ask",
        json={"question": "Who has cloud and Kubernetes experience?", "k": 3},
    )
    assert resp.status_code == 200
    body = resp.json()

    # pre-LLM answer is always present
    assert body["answer_before_llm"]
    assert body["answer_before_llm"].startswith("Top matches for:")

    # post-LLM answer is null without a key
    assert body["answer_after_llm"] is None
    assert body["llm_used"] is None

    # retrieval surfaced relevant candidates
    assert body["retrieved_from"]
    assert {"Diego Fernandez", "Fatima Al-Sayed"} & set(body["retrieved_from"])


def test_ask_validates_k_range():
    resp = client.post("/ask", json={"question": "x", "k": 99})
    assert resp.status_code == 422  # k must be 1..10
