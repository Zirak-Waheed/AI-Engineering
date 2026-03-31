"""Paths, environment, LLM prompts, and shared constants for the RAG practice package."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PRACTICE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = _PRACTICE_DIR / "uploads"
FAISS_CACHE_ROOT = _PRACTICE_DIR / "faiss_index"

load_dotenv(_REPO_ROOT / ".env")

CHUNKS_PER_SUBQUERY = 8
CHUNKS_IN_ANSWER_CONTEXT = 5
RRF_CONSTANT = 60
CHAT_MODEL_NAME = "gpt-4o-mini"

RAG_QUERY_EXPANSION_SYSTEM_PROMPT = (
    "You write short search queries for a vector database. "
    "Given the user question, output exactly 4 alternative queries "
    "that could retrieve relevant passages — one query per line, "
    "no numbering or bullets."
)

RAG_ANSWER_SYSTEM_PROMPT = (
    "Use only the following context to answer. If the answer is not in the "
    "context, say you don't know.\n\nContext:\n{context}"
)

META_FILENAME = "index_meta.json"


def single_index_dir() -> Path:
    """Directory for the single FAISS index (created by POST /index/documents)."""
    return FAISS_CACHE_ROOT / "index"


def rag_query_expansion_prompt_template() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", RAG_QUERY_EXPANSION_SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )


def rag_answer_prompt_template() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", RAG_ANSWER_SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )
