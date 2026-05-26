# Docs RAG: Document Retrieval & Augmented Generation Pipeline

A complete **Retrieval-Augmented Generation (RAG)** solution for document processing, indexing, and retrieval with enhanced response generation using LLMs, fully local.

## 🎯 Key Features

- **Intelligent Ingestion**: Scraping, cleaning, and adaptive chunking of documents
- **Efficient Embeddings**: Embedding generation with `FastEmbed` (CPU-optimized)
- **Vector Storage**: Semantic storage and search with Qdrant
- **RAG Pipeline**: Augmented generation chain with retrieved context
- **RESTful API**: FastAPI with validated Pydantic schemas
- **Continuous Evaluation**: RAGAS framework for automatic quality assessment
- **Observability**: Langfuse integration for tracing and monitoring
- **Flexibility**: Two Ollama configurations (Docker + Local) to optimize inference speed


## 🛠️ Technology Stack

### Backend & Core
- **FastAPI**: Asynchronous web framework
- **Pydantic**: Data validation and configuration
- **Uvicorn**: ASGI server

### LLM & Embeddings
- **LangChain**: LLM chain orchestration
- **Ollama**: Local models (Gemma4, Nomic Embed Text)
- **FastEmbed**: CPU-optimized embeddings
- **LangChain-Ollama**: Ollama integration

### Vector Database
- **Qdrant**: Scalable vector database
- **Qdrant Client**: Python SDK

### Evaluation & Observability
- **RAGAS**: Automatic metrics for RAG (Retrieval, Faithfulness, Relevance)
- **Langfuse v2**: Tracing, debugging, and LLM call monitoring
- **PostgreSQL**: Database for Langfuse

### Infrastructure
- **Docker**: Containerization
- **Docker Compose**: Multi-service orchestration

## 📋 Prerequisites

- Python ≥ 3.12
- Docker & Docker Compose
- Ollama (see configurations below)
- ~4GB RAM minimum (8GB+ recommended)

## 🚀 Quick Start

### Option 1: Ollama in Docker (Recommended for Production)

**Advantage**: Completely isolated, portable, and reproducible environment.

```bash
# Configure environment variables
cp .env.example .env
# Edit .env and set OLLAMA_MODE=docker

# Start with Ollama in container
source .env
./start.sh
```

**docker-compose.yml** includes:
- ✅ FastAPI App
- ✅ Qdrant vector database
- ✅ Langfuse with PostgreSQL
- ✅ **Ollama in container** (via docker-compose.ollama.yml)

### Option 2: Local Ollama (Recommended for Development)

**Advantage**: Direct GPU access, faster inference.

#### Installation:
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows
# Download from https://ollama.ai
```

#### Run:
```bash
# In a separate terminal
ollama serve

# In another terminal, download models
ollama pull gemma4:e2b-mlx
ollama pull nomic-embed-text
```

#### Launch App:
```bash
cp .env.example .env
# Edit .env and set OLLAMA_MODE=local

source .env
./start.sh
```

**docker-compose.yml** will connect to:
- `http://host.docker.internal:11434` (macOS/Windows)
- `http://host.docker.internal:11434` (Linux with Docker Desktop)

## 📝 Configuration

### Environment Variables (.env)

```bash
# Ollama mode: "docker" or "local"
OLLAMA_MODE=docker

# Ollama base URL (automatically configured by start.sh based on OLLAMA_MODE)
# No need to manually edit if you use start.sh
OLLAMA_BASE_URL=http://ollama:11434

# Langfuse (Observability)
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_DB_PASSWORD=your_secure_password
LANGFUSE_NEXTAUTH_SECRET=your_nextauth_secret
LANGFUSE_SALT=your_salt
LANGFUSE_ENCRYPTION_KEY=your_encryption_key
LANGFUSE_INIT_PASSWORD=admin_password
LANGFUSE_INIT_EMAIL=admin@example.com
```

See `.env.example` for complete template.

## 📖 Project Structure

```
docs_rag/
├── src/
│   ├── api/
│   │   ├── main.py           # Main FastAPI app
│   │   └── schemas.py        # Pydantic models
│   │
│   ├── ingestion/
│   │   ├── scraper.py        # Downloads and cleans documents
│   │   ├── chunker.py        # Text fragmentation strategy
│   │   └── embedder.py       # Embedding generation and storage
│   │
│   ├── retrieval/
│   │   ├── retriever.py      # Semantic search in Qdrant
│   │   └── reranker.py       # Cross-encoder reranking
│   │
│   ├── generation/
│   │   ├── chain.py          # Main RAG chain
│   │   └── prompts.py        # Prompt templates
│   │
│   ├── shared/
│   │   └── ...               # Shared utilities
│   │
│   └── data/
│       └── docs/             # Document storage
│
├── evals/
│   ├── golden_dataset.json   # Reference Q&As for evaluation
│   ├── run_evals.py          # Main evaluation script
│   ├── metrics.py            # RAGAS + custom metrics
│   └── results/              # Results history
│
├── scripts/
│   ├── ingest.py             # Document ingestion pipeline
│   └── generate_dataset.py   # Golden dataset generation with LLM
│
├── docker-compose.yml        # Main multi-service config
├── docker-compose.ollama.yml # Ollama config (composed with main)
├── Dockerfile                # Application image
├── pyproject.toml            # Project dependencies
├── settings.py               # Model configuration
└── start.sh                  # Startup script for Ollama mode
```

## 🔄 Data Flow

```
1. INGESTION
   └─ Documents → Scraper → Chunker → Embedder → Qdrant

2. RETRIEVAL (on each query)
   └─ Query → Embedding → Semantic search → Reranking → Top-K documents

3. GENERATION (on each query)
   └─ Query + Context → Prompt Template → Ollama LLM → Response

4. MONITORING
   └─ All operations → Langfuse (traces, metrics, debugging)
```


## 📊 Configuration Comparison

| Aspect | Docker | Local |
|--------|--------|-------|
| **GPU Access** | ❌ (requires configuration) | ✅ Direct access |
| **Inference Speed** | Medium | ⚡ Fast |
| **Portability** | ✅ Excellent | ❌ OS-dependent |
| **Memory** | Isolated in container | Shared with host |
| **Setup** | Automatic | Manual (ollama serve) |
| **Best for** | Production/CI-CD | Local development |

## 🔌 API Endpoints

```bash
# Health check
GET /health

# Query RAG
POST /query
{
  "question": "What is the architecture?",
  "top_k": 5
}

# Get documents
GET /documents/{doc_id}

# List collections
GET /collections
```

Interactive documentation: http://localhost:8000/docs

## 📈 Evaluation

The project includes a complete evaluation framework with:

- **Retrieval Score**: Relevance of retrieved documents
- **Faithfulness**: Consistency between response and context
- **Answer Relevance**: Response relevance to the query
- **Custom Metrics**: Latency, precision, coverage

```bash
# Run evaluation
python evals/run_evals.py --dataset evals/golden_dataset.json

# Results saved in evals/results/
```



Pasa esto de un readme a ingles: ## Development Workflow

### Running evals (requires Apple Silicon)
\
bash
python evals/run_evals.py
\

### CI

The CI pipeline runs linting, type checking, unit tests, and a smoke test using an LLM mock. Full evals are run locally before merging into main.


### Current Performance Metrics

Latest evaluation results (2026-05-24):

Generation model: "gemma4:e2b-mlx"
Chunk size: 700
Chunk overlap: 100
Chunk min size: 100
Embedding model: "nomic-embed-text"

| Metric | Baseline Score | With Reranker Score |
|--------|-------|-------|
| **Faithfulness** | 91.86% | 93.01% |
| **Answer Relevancy** | 79.55% | 84.27% |
| **Context Precision** | 54.76% | 77.35% |
| **Context Recall** | 61.90% | 66.67% |

- **Faithfulness**: High accuracy in response grounding to context
- **Answer Relevancy**: Strong alignment between answers and queries
- **Context Precision**: Good relevance of retrieved documents
- **Context Recall**: Room for improvement in capturing all relevant documents
