from __future__ import annotations

from app.config.settings import settings


class PromptBuilder:
    """Constructs a grounded prompt using the retrieved context.

    The important idea is that the LLM sees only the relevant chunks plus clear
    instructions, rather than the entire document or a vague prompt.
    """

    @staticmethod
    def build(
        question: str, retrieved_chunks: list[dict], top_k: int | None = None
    ) -> str:
        effective_top_k = settings.top_k if top_k is None else top_k
        context_blocks = []
        for index, chunk in enumerate(retrieved_chunks[:effective_top_k], start=1):
            metadata = chunk.get("metadata", {})
            source = metadata.get("filename") or "unknown-document"
            pages = metadata.get("page_ids") or [metadata.get("page_number", "?")]
            page = ", ".join(str(page_id) for page_id in pages)
            text = chunk.get("text", "").strip()
            context_blocks.append(f"[Source {index}] {source} | page {page}\n{text}")

        context_text = "\n\n---\n\n".join(context_blocks)

        return f"""You are answering a question using only the provided context.

Answer using the supplied context only.
Do not invent facts.
If the answer is not present in the context, say clearly that the information is not available in the supplied material.
When possible, mention the source document and page numbers.

Question:
{question}

Relevant retrieved context:
{context_text}

Answer:
"""
