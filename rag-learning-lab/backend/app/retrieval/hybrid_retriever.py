from __future__ import annotations

from typing import Any

from app.retrieval.base import RetrieverBase
from app.retrieval.rrf import fuse_results


class HybridRetriever(RetrieverBase):
    """Hybrid retriever combining BM25 and semantic search via Reciprocal Rank Fusion.

    This retriever runs both BM25 and semantic search, then fuses their rankings
    using RRF to produce a single, intelligently combined result set.
    """

    def __init__(
        self,
        bm25_retriever: RetrieverBase,
        semantic_retriever: RetrieverBase,
        rrf_k_constant: int = 60,
        candidate_pool_factor: float = 1.5,
    ):
        """Initialize hybrid retriever.

        Args:
            bm25_retriever: BM25Retriever instance
            semantic_retriever: SemanticRetriever instance
            rrf_k_constant: RRF k parameter (default 60)
            candidate_pool_factor: Multiplier for retrieving candidates before fusion
                                   (default 1.5x top_k). Allows docs not in immediate
                                   top-k of one method to surface if ranked high in other.
        """
        self.bm25_retriever = bm25_retriever
        self.semantic_retriever = semantic_retriever
        self.rrf_k_constant = rrf_k_constant
        self.candidate_pool_factor = candidate_pool_factor

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Search using both BM25 and semantic, then fuse results with RRF.

        Args:
            query: The search query
            top_k: Number of top results to return (after fusion)

        Returns:
            List of fused result dicts with RRF scores
        """
        # Retrieve candidate pool from both methods
        # Use a larger pool to allow cross-method improvements
        candidate_pool_size = max(int(top_k * self.candidate_pool_factor), top_k)

        bm25_results = self.bm25_retriever.search(query, top_k=candidate_pool_size)
        semantic_results = self.semantic_retriever.search(query, top_k=candidate_pool_size)

        # Fuse using RRF
        fused_results = fuse_results(
            result_sets={
                "bm25": bm25_results,
                "semantic": semantic_results,
            },
            k_constant=self.rrf_k_constant,
            top_k=top_k,
        )

        # Wrap with consistent format
        wrapped = []
        for result in fused_results:
            wrapped.append(
                {
                    **result,
                    "retrieval_score": result.get("rrf_score"),
                    "retrieval_method": "hybrid",
                }
            )
        return wrapped

    def index(self, items: list[dict[str, Any]]) -> None:
        """Index documents in both BM25 and semantic retrievers.

        Args:
            items: List of chunk dicts with 'chunk_id', 'text', 'metadata'
        """
        self.bm25_retriever.index(items)
        self.semantic_retriever.index(items)

    def delete_by_document_id(self, document_id: str) -> None:
        """Delete all chunks for a document from both retrievers.

        Args:
            document_id: The document to remove
        """
        self.bm25_retriever.delete_by_document_id(document_id)
        self.semantic_retriever.delete_by_document_id(document_id)
