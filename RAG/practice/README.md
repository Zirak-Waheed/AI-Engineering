# RAG practice API

FastAPI service for PDF indexing (FAISS) and **RAG Fusion** (multi-query retrieval + reciprocal rank fusion, then grounded answers). Interactive API docs: **Swagger UI** at `/docs`, OpenAPI schema at `/openapi.json`.

## Package layout

| File | Responsibility |
|------|----------------|
| `app.py` | Re-exports the ASGI `app` from `api.py` for `uvicorn RAG.practice.app:app` |
| `api.py` | Pydantic models, embedding/RAG cache, upload response mapping, FastAPI app, and all routes |
| `rag_config.py` | Paths, `.env` load, single index directory helper, RAG Fusion prompts and model constants |
| `rag_chunking.py` | `ChunkingParams`, text splitters |
| `rag_indexing.py` | PDF → documents, FAISS build/save/load, uploads, `IndexUploadResult` |
| `rag_fusion.py` | RRF, `RagFusionEngine`, `create_rag_engine` |

## Prerequisites

- Python 3.12+ (recommended)
- An OpenAI API key for embeddings and chat completions

## Setup

From the **repository root** (`AI-Engineering`):

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Create or update `.env` at the repository root with:

```env
OPENAI_API_KEY=sk-...
```

## Run the server

From the **repository root**:

```bash
python3 -m uvicorn RAG.practice.app:app --host 127.0.0.1 --port 8000
```

Equivalent using the package module (same Uvicorn app string under the hood):

```bash
python3 -m RAG.practice --host 127.0.0.1 --port 8000
```

With reload during development:

```bash
python3 -m uvicorn RAG.practice.app:app --host 127.0.0.1 --port 8000 --reload
```

Or:

```bash
python3 -m RAG.practice --host 127.0.0.1 --port 8000 --reload
```

Then open:

- **Swagger UI:** http://127.0.0.1:8000/docs  
- **ReDoc:** http://127.0.0.1:8000/redoc  

## Main endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Liveness check |
| `POST` | `/index/documents` | Upload a PDF to build or replace the single FAISS index (chunking options via form fields) |
| `POST` | `/query` | Ask a question (requires an index from `/index/documents` first; otherwise HTTP 404) |

The vector index lives under `faiss_index/index/`; uploaded PDFs are stored under `uploads/` (both paths are gitignored).
