from __future__ import annotations

from app.embeddings.base import EmbeddingProvider
from app.retrieval.base import RetrieverBase
from app.retrieval.bm25_index import BM25Index
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.semantic_retriever import SemanticRetriever
from app.vectorstore.chroma_store import SimpleVectorStore


def create_retriever(
    method: str,
    vector_store: SimpleVectorStore,
    embedding_provider: EmbeddingProvider,
    bm25_index: BM25Index,
    rrf_k_constant: int = 60,
    candidate_pool_factor: float = 1.5,
) -> RetrieverBase:
    """Factory for creating retriever instances by method name.

    Follows the established pattern from embeddings/factory.py and chunker_factory.py.
    Normalizes method names (lowercase, underscore→dash) and validates supported methods.

    Args:
        method: The retrieval method ("bm25", "semantic", or "hybrid")
        vector_store: SimpleVectorStore instance for semantic retrieval
        embedding_provider: EmbeddingProvider for embedding queries/documents
        bm25_index: BM25Index instance for keyword retrieval
        rrf_k_constant: RRF k parameter for hybrid retrieval (default 60)
        candidate_pool_factor: Multiplier for candidate pool in hybrid (default 1.5)

    Returns:
        A RetrieverBase implementation matching the requested method

    Raises:
        ValueError: If method is not one of the supported options
    """
    # Normalize method name (lowercase, strip whitespace, underscore→dash)
    normalized = method.strip().lower().replace("_", "-")

    if normalized in {"semantic", "semantic-search"}:
        return SemanticRetriever(
            vector_store=vector_store,
            embedding_provider=embedding_provider,
        )

    if normalized in {"bm25", "bm-25"}:
        return BM25Retriever(bm25_index=bm25_index)

    if normalized == "hybrid":
        # Create both underlying retrievers for hybrid
        bm25_retriever = BM25Retriever(bm25_index=bm25_index)
        semantic_retriever = SemanticRetriever(
            vector_store=vector_store,
            embedding_provider=embedding_provider,
        )
        return HybridRetriever(
            bm25_retriever=bm25_retriever,
            semantic_retriever=semantic_retriever,
            rrf_k_constant=rrf_k_constant,
            candidate_pool_factor=candidate_pool_factor,
        )

    raise ValueError(
        f"Unsupported retrieval method: {method!r}. "
        f"Supported methods: bm25, semantic, hybrid"
    )
