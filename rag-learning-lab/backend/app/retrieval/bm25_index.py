from __future__ import annotations

import re
from typing import Any

from rank_bm25 import BM25Okapi


class BM25Index:
    """BM25 keyword-based retrieval index for comparison with vector search.

    This index runs independently of the vector store and uses keyword matching
    via BM25 to retrieve relevant chunks. Unlike cosine similarity (which uses
    dense embeddings), BM25 operates on tokenized text and is useful for teaching
    how different retrieval methods rank documents differently.

    The index is rebuilt whenever documents are added or deleted, ensuring it
    always reflects the current chunk corpus.
    """

    def __init__(self):
        self.corpus: list[str] = []
        self.items: list[dict[str, Any]] = []
        self.bm25: BM25Okapi | None = None

    def build(self, items: list[dict[str, Any]]) -> None:
        """Rebuild the BM25 index from the current set of chunks.

        Args:
            items: List of chunk dicts from vector_store._items, each with
                   'chunk_id', 'text', 'metadata', etc.
        """
        self.items = items
        self.corpus = [self._tokenize(item.get("text", "")) for item in items]
        self.bm25 = BM25Okapi(self.corpus)

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Search for chunks using BM25 keyword matching.

        Args:
            query: User's question or search text
            top_k: Number of top results to return

        Returns:
            List of dicts with same structure as vector_store.query() results,
            but with 'bm25_score' instead of 'similarity_score'.
        """
        if not self.bm25 or not self.items:
            return []

        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # Pair scores with items
        scored: list[tuple[float, int]] = [
            (score, idx) for idx, score in enumerate(scores)
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)

        # Build results in same structure as vector store
        results = []
        for score, idx in scored[:top_k]:
            item = self.items[idx]
            result = {
                **item,
                "bm25_score": round(float(score), 1),
            }
            results.append(result)

        return results

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple tokenization: lowercase and split on word boundaries.

        This is intentionally simple and visible for teaching purposes.
        BM25 quality is sensitive to tokenization choices, so keeping this
        explicit and configurable is important for a learning project.

        Args:
            text: Raw text to tokenize

        Returns:
            List of lowercase tokens
        """
        # Split on word boundaries, keep only alphanumeric + underscore
        tokens = re.findall(r"\b\w+\b", text.lower())
        return tokens
