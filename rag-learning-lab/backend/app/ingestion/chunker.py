from __future__ import annotations

from typing import List


def split_text_into_chunks(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be greater than or equal to 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    words = text.split()
    if not words:
        return []

    step = chunk_size - chunk_overlap
    chunks: List[str] = []

    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk_words = words[start:end]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
    for i in range(len(chunks)):
        print(i, chunks[i], flush=True)
    return chunks


def chunk_document_pages(
    pages: list[dict], chunk_size: int, chunk_overlap: int
) -> list[dict]:
    chunks: list[dict] = []

    for page in pages:
        page_text = page.get("text", "")
        split_chunks = split_text_into_chunks(
            page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        for idx, chunk_text in enumerate(split_chunks):
            chunks.append(
                {
                    "document_id": page.get("document_id"),
                    "filename": page.get("filename"),
                    "page_number": page.get("page_number"),
                    "chunk_id": f"{page.get('page_number')}_{idx}",
                    "chunk_text": chunk_text,
                }
            )

    return chunks
