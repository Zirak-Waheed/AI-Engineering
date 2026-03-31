"""RAG Fusion: multi-query expansion, RRF over retrievals, and grounded answers."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .rag_config import (
    CHAT_MODEL_NAME,
    CHUNKS_IN_ANSWER_CONTEXT,
    CHUNKS_PER_SUBQUERY,
    RRF_CONSTANT,
    rag_answer_prompt_template,
    rag_query_expansion_prompt_template,
)
from .rag_indexing import load_faiss_index


def _parse_expansion_lines(message_content: object) -> list[str]:
    text = str(message_content)
    return [line.strip() for line in text.splitlines() if line.strip()][:4]


def _deduplicate_search_queries(question: str, llm_lines: list[str]) -> list[str]:
    deduplicated: list[str] = []
    seen: set[str] = set()
    for candidate in [question, *llm_lines]:
        if candidate and candidate not in seen:
            seen.add(candidate)
            deduplicated.append(candidate)
    return deduplicated


def reciprocal_rank_fusion_top_k(
    retrieval_rankings_per_query: list[list[Document]],
    *,
    rrf_constant: int,
    top_k: int,
) -> list[Document]:
    chunk_rrf_score_by_text: dict[str, float] = {}
    representative_chunk_by_text: dict[str, Document] = {}
    for ranked_chunks in retrieval_rankings_per_query:
        for rank, chunk in enumerate(ranked_chunks, start=1):
            chunk_text = chunk.page_content
            contribution = 1.0 / (rrf_constant + rank)
            chunk_rrf_score_by_text[chunk_text] = (
                chunk_rrf_score_by_text.get(chunk_text, 0.0) + contribution
            )
            representative_chunk_by_text.setdefault(chunk_text, chunk)

    chunks_sorted = sorted(
        representative_chunk_by_text.values(),
        key=lambda doc: chunk_rrf_score_by_text[doc.page_content],
        reverse=True,
    )
    return chunks_sorted[:top_k]


def _unique_source_paths(chunks: list[Document]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        source = chunk.metadata.get("source")
        if isinstance(source, str) and source not in seen:
            seen.add(source)
            unique.append(source)
    return unique


@dataclass(frozen=True)
class RagQueryResult:
    answer: str
    search_queries: tuple[str, ...]
    source_paths: tuple[str, ...]


class RagFusionEngine:
    """RAG Fusion: multi-query retrieval + RRF, then grounded answer generation."""

    def __init__(
        self,
        vector_store: FAISS,
        chat_model: ChatOpenAI,
        *,
        chunks_per_subquery: int = CHUNKS_PER_SUBQUERY,
        chunks_in_context: int = CHUNKS_IN_ANSWER_CONTEXT,
        rrf_constant: int = RRF_CONSTANT,
    ) -> None:
        self._vector_store = vector_store
        self._chat_model = chat_model
        self._chunks_per_subquery = chunks_per_subquery
        self._chunks_in_context = chunks_in_context
        self._rrf_constant = rrf_constant
        self._query_expansion_template = rag_query_expansion_prompt_template()
        self._answer_prompt = rag_answer_prompt_template()

    def query(self, question: str) -> RagQueryResult:
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty.")

        expansion_message = self._chat_model.invoke(
            self._query_expansion_template.format_messages(question=question)
        )
        llm_lines = _parse_expansion_lines(expansion_message.content)
        deduplicated_search_queries = _deduplicate_search_queries(question, llm_lines)

        retrieval_rankings = [
            self._vector_store.similarity_search(q, k=self._chunks_per_subquery)
            for q in deduplicated_search_queries
        ]
        top_chunks = reciprocal_rank_fusion_top_k(
            retrieval_rankings,
            rrf_constant=self._rrf_constant,
            top_k=self._chunks_in_context,
        )
        fused_context = "\n\n".join(chunk.page_content for chunk in top_chunks)

        answer = (self._answer_prompt | self._chat_model | StrOutputParser()).invoke(
            {"context": fused_context, "question": question}
        )
        unique_sources = _unique_source_paths(top_chunks)

        return RagQueryResult(
            answer=answer,
            search_queries=tuple(deduplicated_search_queries),
            source_paths=tuple(unique_sources),
        )


def create_rag_engine(
    *,
    embedding_model: OpenAIEmbeddings | None = None,
) -> RagFusionEngine:
    """Load the single FAISS index and wrap it with RAG Fusion + chat model."""
    resolved_embeddings = embedding_model or OpenAIEmbeddings()
    vector_store = load_faiss_index(resolved_embeddings)
    chat_model = ChatOpenAI(model=CHAT_MODEL_NAME, temperature=0)
    return RagFusionEngine(vector_store, chat_model)
