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
    document_words: list[str] = []
    word_pages: list[int] = []
    for page in non_empty_pages:
        page_words = page["text"].split()
        document_words.extend(page_words)
        word_pages.extend([page.get("page_number")] * len(page_words))

    split_chunks = chunker.chunk(document_text)
    chunks: list[dict] = []
    search_start = 0
    overlap_words = chunk_overlap if strategy.lower() in {"fixed", "recursive"} else 0

    for chunk_index, chunk_text in enumerate(split_chunks):
        chunk_words = chunk_text.split()
        start = _find_chunk_start(document_words, chunk_words, search_start)
        if start is None:
            start = _find_chunk_start(document_words, chunk_words, 0) or 0
        end = min(start + len(chunk_words), len(word_pages))
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
                "chunk_text": chunk_text,
                "chunking_strategy": strategy,
                "chunk_size": chunk_size,
                "chunk_index": chunk_index,
            }
        )
        search_start = max(start + len(chunk_words) - overlap_words, start + 1)

    return chunks


def _find_chunk_start(
    document_words: list[str], chunk_words: list[str], search_start: int
) -> int | None:
    if not chunk_words:
        return search_start
    last_start = len(document_words) - len(chunk_words)
    for start in range(search_start, last_start + 1):
        if document_words[start : start + len(chunk_words)] == chunk_words:
            return start
    return None
