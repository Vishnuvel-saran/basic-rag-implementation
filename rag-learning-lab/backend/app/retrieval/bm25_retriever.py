from __future__ import annotations

from typing import Any

from app.retrieval.base import RetrieverBase
from app.retrieval.bm25_index import BM25Index


class BM25Retriever(RetrieverBase):
    """BM25-based keyword retriever using the BM25 ranking algorithm.

    This retriever uses keyword matching and BM25 scoring to find relevant documents.
    It's useful for queries with specific keywords and can complement semantic search.
    """

    def __init__(self, bm25_index: BM25Index):
        self.bm25_index = bm25_index

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Search for documents using BM25 keyword matching.

        Args:
            query: The search query (tokenized internally by BM25)
            top_k: Number of top results to return

        Returns:
            List of result dicts with BM25 scores
        """
        results = self.bm25_index.search(query, top_k=top_k)

        # Wrap results with consistent format
        wrapped = []
        for result in results:
            wrapped.append(
                {
                    **result,
                    "retrieval_score": result.get("bm25_score"),
                    "retrieval_method": "bm25",
                }
            )
        return wrapped

    def index(self, items: list[dict[str, Any]]) -> None:
        """Index documents for BM25 keyword search.

        Args:
            items: List of chunk dicts with 'chunk_id', 'text', 'metadata'
        """
        # BM25Index.build() expects the full items structure
        self.bm25_index.build(items)

    def delete_by_document_id(self, document_id: str) -> None:
        """Delete all chunks for a document.

        Note: BM25Index doesn't have a delete operation; it gets rebuilt
        with only the remaining documents. This is handled at the retriever
        factory/pipeline level by rebuilding with filtered items.

        Args:
            document_id: The document to remove (handled externally)
        """
        # BM25Index will be rebuilt by the pipeline after deletion from vector store
        pass
