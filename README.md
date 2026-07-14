# simple-rag — CV Screener + RAG Q&A (LlamaIndex + LangChain + LangGraph)

A small, runnable example that generates 10 sample CV PDFs and answers questions
about the candidates.

## Architecture — who does what

| Layer | Library | Responsibility |
|-------|---------|----------------|
| **Retrieval** | **LlamaIndex** | Reads the CV PDFs, chunks + embeds them, stores vectors in Chroma, and retrieves the right chunks for a query. |
| **Prompts / models / tools** | **LangChain** | Prompt templates, chat models (OpenRouter / Groq / Claude), and structured-output tool-calling in the screener. |
| **Workflow** | **LangGraph** | Coordinates the steps into a state machine (`retrieve → extractive_answer → llm_answer`). |

It offers two entry points built on those layers:

1. **Screener** (`main.py`) — ranks candidates against a job description.
2. **RAG Q&A bot** (`ask.py` / `app.py`) — answers free-form questions about the
   candidates, grounded in the CV text via retrieval. Exposed as a **FastAPI**
   service that returns two answers per question (pre-LLM and post-LLM).

## Screener — what it does

1. `generate_cvs.py` writes 10 fictional CVs as PDFs to `data/cvs/`.
2. A LangGraph state machine processes each CV through three nodes:

   ```
   START → load_cv → extract_profile → score_candidate → END
   ```

   - **load_cv** — LangChain `PyPDFLoader` reads the PDF into text.
   - **extract_profile** — extracts a structured `CandidateProfile`.
   - **score_candidate** — scores fit (0–100) against the job description.

3. `main.py` runs the graph over every CV and prints a ranked shortlist.

## RAG Q&A bot — what it does

`ask.py` drives a retrieve → generate graph:

```
START → retrieve → generate → END
```

- **retrieve** (LlamaIndex) — `SimpleDirectoryReader` reads the CV PDFs (each
  tagged with its candidate), `SentenceSplitter` chunks them, a local
  HuggingFace model (`all-MiniLM-L6-v2`) embeds them, and the vectors are
  persisted in **Chroma** at `data/chroma/`. LlamaIndex's `VectorStoreIndex`
  retrieves the most similar chunks. Embeddings are local, so retrieval works
  offline with no API key.
- **generate** (LangChain) — the model answers using only the retrieved excerpts
  and cites candidates by name. Without a key, the pre-LLM extractive answer is
  returned and `answer_after_llm` is null.

```bash
python build_index.py             # build the Chroma index (auto-builds on first ask)
python build_index.py --rebuild   # wipe and rebuild after changing CVs

python ask.py                              # interactive REPL
python ask.py "who knows Kubernetes?"      # single question
```

Because it's semantic, a query like *"who can build generative AI chatbots?"*
surfaces the ML/LLM candidates even though those exact words aren't in their CVs.

## FastAPI service

`app.py` exposes the RAG bot over HTTP. The key feature: **`POST /ask` returns
two answers** for the same question —

- `answer_before_llm` — extractive answer straight from Chroma retrieval (no LLM).
- `answer_after_llm` — grounded generation from the configured LLM (or `null`).

```bash
python app.py                       # serves on PORT (default 8000) -> /docs
PORT=8100 python app.py             # pick another port if 8000 is in use
uvicorn app:app --port 8100         # or drive uvicorn directly

curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Who has cloud and Kubernetes experience?", "k": 3}'
```

Endpoints: `GET /health`, `POST /ask`. Interactive docs at `/docs`.

## LLM vs. fallback

The "after LLM" answer uses whichever provider key is set, in priority order:

| Provider | Env var | Default model | Get a key |
|----------|---------|---------------|-----------|
| **OpenRouter** | `OPENROUTER_API_KEY` | `openai/gpt-4o` | https://openrouter.ai/keys |
| **Groq** (free) | `GROQ_API_KEY` | `llama-3.3-70b-versatile` | https://console.groq.com/keys |
| **Claude** | `ANTHROPIC_API_KEY` | `claude-sonnet-4-6` | https://console.anthropic.com/ |

Override the model per provider with `OPENROUTER_MODEL` / `GROQ_MODEL`. Without
any key, retrieval still works fully — the API returns the pre-LLM answer and
`answer_after_llm: null`. Embeddings are always local, so retrieval never needs
a key.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# optional — enables the "after LLM" answer
cp .env.example .env   # then add OPENROUTER_API_KEY (or GROQ / ANTHROPIC)
```

## Testing

```bash
pytest -q      # 8 tests: retrieval, RAG graph, and API endpoints
```

Tests run without any API key (the LLM path is asserted to be null in fallback
mode), so they're hermetic and fast.

## Run

```bash
python generate_cvs.py          # create the 10 sample CV PDFs
python main.py                  # screen against the built-in job description
python main.py "your job text"  # or supply your own
```

## Layout

```
generate_cvs.py       # PDF generation (reportlab)
main.py               # screener entry point: run graph + rank
ask.py                # RAG Q&A CLI (REPL or single question)
app.py                # FastAPI service — GET /health, POST /ask
build_index.py        # build/rebuild the Chroma vector index
cv_screener/
  graph.py            # screener LangGraph wiring
  nodes.py            # load / extract / score nodes (+ heuristic fallback)
  rag.py              # LlamaIndex retrieval: reader + splitter + embed + Chroma
  rag_graph.py        # RAG LangGraph: retrieve → extractive_answer → llm_answer
  schemas.py          # Pydantic models for structured output
  state.py            # graph state (TypedDict)
  llm.py              # LLM factory (Groq / Claude) + key detection
tests/
  conftest.py         # fixtures: hermetic (no-key) + index bootstrap
  test_rag.py         # retrieval + RAG graph tests
  test_api.py         # FastAPI endpoint tests
```
