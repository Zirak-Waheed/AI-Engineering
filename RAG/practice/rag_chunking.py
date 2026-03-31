"""Document chunking strategies and LangChain splitter construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter


@dataclass(frozen=True)
class ChunkingParams:
    """How to split documents before embedding."""

    strategy: Literal["recursive", "character"]
    chunk_size: int
    chunk_overlap: int

    def validate(self) -> None:
        if self.chunk_size < 50 or self.chunk_size > 32000:
            raise ValueError("chunk_size must be between 50 and 32000.")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and < chunk_size.")


DEFAULT_CHUNKING = ChunkingParams(
    strategy="recursive",
    chunk_size=500,
    chunk_overlap=50,
)


def get_text_splitter(
    params: ChunkingParams,
) -> RecursiveCharacterTextSplitter | CharacterTextSplitter:
    params.validate()
    if params.strategy == "recursive":
        return RecursiveCharacterTextSplitter(
            chunk_size=params.chunk_size,
            chunk_overlap=params.chunk_overlap,
        )
    if params.strategy == "character":
        return CharacterTextSplitter(
            chunk_size=params.chunk_size,
            chunk_overlap=params.chunk_overlap,
            separator="\n",
        )
    raise ValueError(f"Unknown chunking strategy: {params.strategy}")
