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
    non_empty_pages = [page for page in pages if page.get("text", "").strip()]
    if not non_empty_pages:
        return []

    document_text = "\n\n".join(page["text"].strip() for page in non_empty_pages)

    # word_pages[i] tells us which page word i of document_text belongs to.
    # Built once, up front, from ground truth -- not reconstructed later.
    word_pages: list[int] = []
    for page in non_empty_pages:
        page_words = page["text"].split()
        word_pages.extend([page.get("page_number")] * len(page_words))

    spans = chunker.chunk_with_offsets(document_text)
    chunks: list[dict] = []

    for chunk_index, span in enumerate(spans):
        start = max(span.start, 0)
        end = min(span.end, len(word_pages))
        page_ids = list(dict.fromkeys(word_pages[start:end]))
        source_page = non_empty_pages[0]
        if page_ids:
            source_page = next(
                page
                for page in non_empty_pages
                if page.get("page_number") == page_ids[0]
            )

        chunks.append(
            {
                "document_id": source_page.get("document_id"),
                "filename": source_page.get("filename"),
                "page_number": page_ids[0]
                if page_ids
                else source_page.get("page_number"),
                "page_ids": page_ids,
                "chunk_id": f"chunk_{chunk_index + 1:03d}",
                "chunk_text": span.text,
                "chunking_strategy": strategy,
                "chunk_size": chunk_size,
                "chunk_index": chunk_index,
            }
        )

    return chunks
