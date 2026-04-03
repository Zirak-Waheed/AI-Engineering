# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is an AI Engineering learning repository containing exploratory notebooks and working implementations for LangChain, RAG systems, and AI agents.

## Common Commands

### Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### RAG Practice API
Run the FastAPI server for PDF indexing and RAG Fusion queries:
```bash
python3 -m uvicorn RAG.practice.app:app --host 127.0.0.1 --port 8000
# With reload for development:
python3 -m uvicorn RAG.practice.app:app --host 127.0.0.1 --port 8000 --reload
```

### GAIA Agent
Run the GAIA benchmark agent (requires GAIA dataset files):
```bash
cd AGENTS/GAIA
pip install -r requirements.txt
python3 app.py      # Gradio UI
python3 evaluate.py # Run evaluation
```

## Architecture

### RAG System (`RAG/practice/`)
FastAPI service implementing RAG Fusion with multi-query retrieval and reciprocal rank fusion:
- `api.py` - FastAPI routes: `/health`, `/index/documents` (PDF upload), `/query`
- `rag_fusion.py` - `RagFusionEngine` class: query expansion via LLM, parallel similarity searches across sub-queries, RRF scoring to combine results
- `rag_indexing.py` - FAISS vector store creation from PDF documents
- `rag_chunking.py` - Text splitting with configurable chunk size/overlap

### GAIA Agent (`AGENTS/GAIA/`)
LangGraph-based ReAct agent for the GAIA benchmark (achieved 70% on 20 questions):
- `agent.py` - Core agent with `create_react_agent`, answer normalization, and verification loop
- `tools.py` - Tool definitions: `tavily_search`, `visit_webpage`, `python_repl`, `read_file`, `analyze_image`, `transcribe_audio`, `get_youtube_transcript`
- `app.py` - Gradio web interface
- `evaluate.py` / `submit.py` - Benchmark evaluation scripts

### CrewAI (`AGENTS/crew-ai/`)
CrewAI multi-agent setup with YAML-based agent/task configuration in `config/`.

### LangChain Notebooks (`LANG-CHAIN/`)
- `Model_prompt_parser.ipynb` - Prompt templates and output parsing
- `Multi-Chains.ipynb` - Chain composition patterns

## Environment Variables

Required in `.env` at repository root:
```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=...  # For GAIA agent web search
```

## Key Dependencies

- `langchain`, `langchain-openai`, `langgraph` - LLM orchestration
- `faiss-cpu` - Vector similarity search
- `fastapi`, `uvicorn` - API server
- `crewai`, `crewai_tools` - Multi-agent framework
- `smolagents[litellm]` - Lightweight agents
