"""FAISS index build, persistence, and upload pipeline (single shared index)."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from .rag_chunking import ChunkingParams, get_text_splitter
from .rag_config import META_FILENAME, UPLOADS_DIR, single_index_dir

NO_INDEX_YET_MESSAGE = (
    "No vector index found. Upload a PDF with POST /index/documents before querying."
)


def _documents_from_pdf(pdf_path: Path) -> list[Document]:
    absolute = str(pdf_path.resolve())
    return [
        Document(
            page_content=page.page_content,
            metadata={**page.metadata, "source": absolute},
        )
        for page in PyPDFLoader(str(pdf_path)).load()
    ]


def _write_meta(index_dir: Path, params: ChunkingParams, sources: list[str]) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "chunking_strategy": params.strategy,
        "chunk_size": params.chunk_size,
        "chunk_overlap": params.chunk_overlap,
        "sources": sources,
    }
    (index_dir / META_FILENAME).write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def load_faiss_index(embedding_model: OpenAIEmbeddings | None = None) -> FAISS:
    """Load the single persisted index. Raises FileNotFoundError if none exists."""
    index_dir = single_index_dir()
    if not (index_dir / "index.faiss").is_file():
        raise FileNotFoundError(NO_INDEX_YET_MESSAGE)
    meta_path = index_dir / META_FILENAME
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing {META_FILENAME} under {index_dir}.")
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    ChunkingParams(
        strategy=raw["chunking_strategy"],
        chunk_size=int(raw["chunk_size"]),
        chunk_overlap=int(raw["chunk_overlap"]),
    ).validate()
    emb = embedding_model or OpenAIEmbeddings()
    return FAISS.load_local(
        str(index_dir),
        emb,
        allow_dangerous_deserialization=True,
    )


@dataclass(frozen=True)
class IndexUploadResult:
    chunk_count: int
    storage_path: str
    chunking: ChunkingParams


def persist_upload_and_index(
    file_bytes: bytes,
    original_filename: str,
    params: ChunkingParams,
    embedding_model: OpenAIEmbeddings | None = None,
) -> IndexUploadResult:
    """Save upload under ``uploads/``, then build or replace the FAISS index."""
    if not original_filename.lower().endswith(".pdf"):
        raise ValueError("Only PDF uploads are supported.")
    params.validate()

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOADS_DIR / Path(original_filename).name
    dest.write_bytes(file_bytes)

    index_dir = single_index_dir()
    if index_dir.is_dir():
        shutil.rmtree(index_dir)

    documents = _documents_from_pdf(dest)
    if not documents:
        raise ValueError("No pages could be read from the PDF.")

    emb = embedding_model or OpenAIEmbeddings()
    chunks = get_text_splitter(params).split_documents(documents)
    if not chunks:
        raise ValueError("No text chunks produced; check PDF content and chunking params.")

    index_dir.mkdir(parents=True, exist_ok=True)
    FAISS.from_documents(chunks, emb).save_local(str(index_dir))
    _write_meta(index_dir, params, [str(dest.resolve())])

    return IndexUploadResult(
        chunk_count=len(chunks),
        storage_path=str(index_dir.resolve()),
        chunking=params,
    )
