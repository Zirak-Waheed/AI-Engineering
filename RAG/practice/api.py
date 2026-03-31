"""FastAPI app: upload one PDF to index, then query with RAG Fusion."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field

from .rag_chunking import ChunkingParams
from .rag_fusion import RagFusionEngine, create_rag_engine
from .rag_indexing import IndexUploadResult, persist_upload_and_index

MAX_UPLOAD_BYTES = 50 * 1024 * 1024

_embedding_model: OpenAIEmbeddings | None = None
_rag_engine: RagFusionEngine | None = None


def _get_embedding_model() -> OpenAIEmbeddings:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = OpenAIEmbeddings()
    return _embedding_model


def _get_rag_engine() -> RagFusionEngine:
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = create_rag_engine(embedding_model=_get_embedding_model())
    return _rag_engine


def _invalidate_rag_engine() -> None:
    global _rag_engine
    _rag_engine = None


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=8000)


class QueryResponse(BaseModel):
    answer: str
    search_queries: list[str]
    source_paths: list[str]


class ChunkingResponse(BaseModel):
    strategy: Literal["recursive", "character"]
    chunk_size: int
    chunk_overlap: int


class IndexUploadResponse(BaseModel):
    chunk_count: int
    storage_path: str
    chunking: ChunkingResponse


@asynccontextmanager
async def _lifespan(_: FastAPI):
    yield
    global _embedding_model, _rag_engine
    _rag_engine = None
    _embedding_model = None


app = FastAPI(
    title="RAG Practice API",
    description="Index one PDF (POST /index/documents), then query (POST /query).",
    version="1.2.0",
    lifespan=_lifespan,
)


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse, tags=["rag"])
async def query_rag(body: QueryRequest) -> QueryResponse:
    try:
        result = _get_rag_engine().query(body.question)
    except FileNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    return QueryResponse(
        answer=result.answer,
        search_queries=list(result.search_queries),
        source_paths=list(result.source_paths),
    )


@app.post("/index/documents", response_model=IndexUploadResponse, tags=["indexing"])
async def index_documents(
    file: Annotated[UploadFile, File(description="PDF to chunk and embed.")],
    chunking_strategy: Annotated[
        Literal["recursive", "character"],
        Form(),
    ] = "recursive",
    chunk_size: Annotated[int, Form(ge=50, le=32000)] = 500,
    chunk_overlap: Annotated[int, Form(ge=0)] = 50,
) -> IndexUploadResponse:
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"File exceeds maximum size of {MAX_UPLOAD_BYTES} bytes.",
        )

    params = ChunkingParams(
        strategy=chunking_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    try:
        result: IndexUploadResult = persist_upload_and_index(
            raw,
            file.filename or "upload.pdf",
            params,
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    _invalidate_rag_engine()
    _get_rag_engine()

    c = result.chunking
    return IndexUploadResponse(
        chunk_count=result.chunk_count,
        storage_path=result.storage_path,
        chunking=ChunkingResponse(
            strategy=c.strategy,
            chunk_size=c.chunk_size,
            chunk_overlap=c.chunk_overlap,
        ),
    )
