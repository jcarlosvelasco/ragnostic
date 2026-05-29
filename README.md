# docs-rag — LangChain Docs Q&A with Automatic Evaluation

A production-grade **Retrieval-Augmented Generation (RAG)** pipeline for querying LangChain documentation in natural language. Built end-to-end with ingestion, retrieval, generation, evaluation, observability, and CI, running entirely on local models with no external API costs.

> **Why this project?** Most RAG demos stop at "it answers questions". This one measures whether the answers are actually correct, tracks quality over time, and blocks deploys when quality drops.

---

## Results

Latest evaluation — 2026-05-27 · model: `gemma4:e2b-mlx` · chunk size: 600 · overlap: 90

| Metric | Baseline | With Reranker | Δ |
|---|---|---|---|
| **Faithfulness** | 91.86% | **97.15%** | +5,29% |
| **Answer Relevancy** | 79.55% | **84.27%** | +4.72% |
| **Context Precision** | 54.76% | **79.07%** | +24.31% |
| **Context Recall** | 61.90% | **68,29%** | +6,39% |

The reranker had the most impact on **Context Precision** (+24,31%): retrieved chunks became significantly more relevant to the query without changing the embedding model or index.

---

## Architecture

```
INGESTION (offline, batch)
─────────────────────────────────────────────────────────
docs.langchain.com/llms.txt
    │
    ├─ scraper.py       Fetches Markdown pages, filters frontend/api-reference
    ├─ chunker.py       RecursiveCharacterTextSplitter (size=700, overlap=100)
    └─ embedder.py      nomic-embed-text via Ollama → Qdrant (DOT similarity)


QUERY PIPELINE (per request)
─────────────────────────────────────────────────────────
User question
    │
    ├─ Embed query          nomic-embed-text
    ├─ Retrieve k=20        Qdrant semantic search
    ├─ Rerank               cross-encoder/ms-marco-MiniLM-L-6-v2
    ├─ Select top-5         highest reranker score
    └─ Generate             gemma4:e2b via Ollama + LangChain chain


EVALUATION
─────────────────────────────────────────────────────────
golden_dataset.json  (50 Q&A pairs generated from real chunks)
    └─ RAGAS metrics: faithfulness · answer_relevancy · context_precision · context_recall


OBSERVABILITY
─────────────────────────────────────────────────────────
Every request → Langfuse (traces · latency per step · token usage)
```

---

## Engineering Decisions

These are the decisions that had measurable impact, with the data to back them up.

**Reranker: k=20 retrieve → rerank → top-5**
Without reranker: context_precision 54.76%. With CrossEncoder reranking: 79.07%. +24 points for ~30ms extra latency. Worth it.

**Chunk size: 600 tokens, overlap: 90**
Tested 200 (original), 500, and 600. At 200, chunks were too small for the model to generate useful ground truths, questions ended up about code snippets rather than concepts. At 600, chunks contain full ideas and the golden dataset quality improved significantly.

**Scraper filters: excluded `frontend`, `api-reference`, `deepagents` URLs**
These pages contain React components, CSS, and YAML specs, not documentation prose. Including them polluted the vector store with noise that hurt retrieval quality.

**`RecursiveCharacterTextSplitter` over fixed-size character chunking**
Fixed-size chunking truncated mid-word (`"emory.InMemorySaver"`). Recursive splitting respects paragraph → sentence → word hierarchy, producing semantically coherent chunks.

**Embedding model: `nomic-embed-text`**
Best open-source embedding model available via Ollama. 768-dim vectors, fast inference on Apple Silicon, strong retrieval performance on technical documentation.

**LLM: `gemma4:e2b-mlx`**
MLX-optimized for Apple Silicon. Runs at ~40 tokens/s on M1 16GB, fast enough for development and evaluation cycles without API costs.

**Langfuse v2**
Lighter than v3, needs less containers and resources, enough for this use case.

---

## Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Pydantic + Uvicorn |
| UI | Streamlit |
| LLM & Embeddings | Ollama · LangChain · gemma4:e2b · nomic-embed-text |
| Reranking | sentence-transformers · cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Vector Store | Qdrant |
| Evaluation | RAGAS |
| Observability | Langfuse v2 |
| Infrastructure | Docker · Docker Compose |
| CI | GitHub Actions (lint · type check · unit tests · smoke test) |

---

## Project Structure

```
docs-rag/
├── src/
│   ├── api/
│   │   ├── main.py              FastAPI app
│   │   └── schemas.py           Pydantic models
│   ├── ingestion/
│   │   ├── scraper.py           Fetches and filters LangChain docs
│   │   ├── chunker.py           RecursiveCharacterTextSplitter
│   │   └── embedder.py          Generates embeddings and uploads to Qdrant
│   ├── retrieval/
│   │   ├── retriever.py         Semantic search in Qdrant
│   │   └── reranker.py          CrossEncoder reranking
│   ├── generation/
│   │   ├── chain.py             RAG chain (prompt | LLM | parser)
│   │   └── prompts.py           Prompt templates
│   ├── shared/
│   │    ├── langfuse.py         Langfuse callback handler
│   │    ├── vector_store.py     Qdrant utilities
│   └── streamlit_app/
│       └── app.py               Streamlit UI
├── evals/
│   ├── golden_dataset.json      50 Q&A pairs for evaluation
│   ├── run_evals.py             RAGAS evaluation script with thresholds
│   ├── generate_dataset.py      Generates golden dataset from Qdrant chunks
│   └── results/                 Historical eval results
├── tests/unit/                  Unit tests
├── .github/workflows/ci.yml     CI pipeline
├── docker-compose.yml           App + Qdrant + Langfuse
├── docker-compose.ollama.yml    Ollama in Docker (alternative to local)
├── settings.py                  Model and config settings
└── start.sh                     Startup script (selects Ollama mode)
```

---

## Quick Start

### Prerequisites

- Python ≥ 3.12
- Docker & Docker Compose
- Ollama (local) or Docker (for containerized Ollama)

### 1. Clone and configure

```bash
git clone https://github.com/jcarlosvelasco/docs-rag
cd docs-rag
cp .env.example .env
# Edit .env with your Langfuse credentials
```

### 2. Install ollama (only if running locally)
```
  curl -fsSL https://ollama.com/install.sh | sh # for Linux/macOS
  irm https://ollama.com/install.ps1 | iex # for Windows
```

### 3. Start services

**Option A — Local Ollama (recommended for Apple Silicon or fast GPUs, faster inference):**
```bash
ollama serve   # in a separate terminal
# Edit .env: OLLAMA_MODE=local
source .env && ./start.sh
```

**Option B — Ollama in Docker (portable, no local install needed):**
```bash
# Edit .env: OLLAMA_MODE=docker
source .env && ./start.sh
```

### 4. Ingest documents

```bash
python scripts/ingest.py
```

This scrapes `docs.langchain.com`, chunks the Markdown, generates embeddings, and stores them in Qdrant. Takes ~5-10 minutes on first run.

### 5. Query

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How does LangChain manage context?"}'
```

Interactive API docs: http://localhost:8000/docs
Streamlit UI: http://localhost:8501

---

## Evaluation

The project includes a full eval suite using RAGAS with a golden dataset of 50 Q&A pairs generated from real documentation chunks.

```bash

python evals/run_evals.py
```

Evals fail automatically if any metric drops below threshold:

| Metric | Threshold |
|---|---|
| Faithfulness | 0.5 |
| Answer Relevancy | 0.5 |
| Context Precision | 0.5 |
| Context Recall | 0.5 |

Results are saved to `evals/results/latest.json` for historical tracking.

**Generating a new golden dataset:**
```bash
python evals/generate_dataset.py --n-chunks 80

```

---

## Observability

All requests are traced in Langfuse with full visibility into each pipeline step: retrieval latency, reranker scores, LLM tokens used, and end-to-end response time.

To access the Langfuse dashboard: http://localhost:3000

Configure credentials in `.env`:
```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

---

## Development Workflow

### CI

The CI pipeline runs on every push and PR to `main`:

1. **Lint** — `ruff check src/`
2. **Type check** — `mypy src/`
3. **Unit tests** — `pytest tests/unit/`
4. **Smoke test** — starts the API with a mock LLM and verifies the endpoint responds correctly


```bash
python evals/run_evals.py
```

### Adding a new feature

1. Create a branch
2. Make changes
3. Run evals locally: verify no metric drops below threshold
4. Push: CI runs lint, type check, unit tests, and smoke test
5. Merge if CI passes

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `OLLAMA_MODE` | `local` or `docker` | `local` |
| `OLLAMA_BASE_URL` | Ollama API URL | `http://host.docker.internal:11434` |
| `LLM_PROVIDER` | `ollama` or `mock` (CI) | `ollama` |
| `QDRANT_URL` | Qdrant URL | `http://localhost:6333` |
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key | — |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key | — |

See `.env.example` for the full list.

## License

Check [LICENSE.md](LICENSE.md)
