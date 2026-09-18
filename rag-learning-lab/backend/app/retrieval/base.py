from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class RetrieverBase(ABC):
    """Abstract base class for all retrieval strategies.

    Defines a consistent interface for different retrieval methods (BM25, semantic, hybrid).
    All retrievers return results in a uniform structure for easy composition and comparison.
    """

    @abstractmethod
    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Retrieve documents matching the query.

        Args:
            query: The search query (raw text for BM25, will be embedded for semantic)
            top_k: Maximum number of results to return

        Returns:
            List of result dicts with at least:
            - chunk_id: str
            - text: str
            - metadata: dict
            - retrieval_score: float (method-specific score)
            - retrieval_method: str ("bm25", "semantic", or "hybrid")
            - vector: list[float] (optional, typically for semantic)
        """
        pass

    @abstractmethod
    def index(self, items: list[dict[str, Any]]) -> None:
        """Index a batch of documents/chunks.

        Args:
            items: List of chunk dicts with 'chunk_id', 'text', 'metadata', etc.
        """
        pass

    @abstractmethod
    def delete_by_document_id(self, document_id: str) -> None:
        """Delete all chunks belonging to a document.

        Args:
            document_id: The document to remove
        """
        pass
