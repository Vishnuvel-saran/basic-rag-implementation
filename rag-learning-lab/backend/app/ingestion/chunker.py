from __future__ import annotations

from typing import List

from app.embeddings.base import EmbeddingProvider
from app.ingestion.chunker_factory import ChunkerFactory
from app.ingestion.chunkers import ChunkingConfig


def split_text_into_chunks(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    return ChunkerFactory.create(
        "fixed", ChunkingConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    ).chunk(text)


def chunk_document_pages(
    pages: list[dict],
    chunk_size: int,
    chunk_overlap: int,
    strategy: str = "fixed",
    semantic_threshold: float = 0.75,
    embedding_provider: EmbeddingProvider | None = None,
) -> list[dict]:
    chunker = ChunkerFactory.create(
        strategy,
        ChunkingConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            semantic_threshold=semantic_threshold,
        ),
        embedding_provider=embedding_provider,
    )
    chunks: list[dict] = []
    chunk_index = 0

    for page in pages:
        page_text = page.get("text", "")
        split_chunks = chunker.chunk(page_text)

        for idx, chunk_text in enumerate(split_chunks):
            chunks.append(
                {
                    "document_id": page.get("document_id"),
                    "filename": page.get("filename"),
                    "page_number": page.get("page_number"),
                    "chunk_id": f"chunk_{chunk_index + 1:03d}",
                    "chunk_text": chunk_text,
                    "chunking_strategy": strategy,
                    "chunk_size": chunk_size,
                    "chunk_index": chunk_index,
                }
            )
            chunk_index += 1

    return chunks
