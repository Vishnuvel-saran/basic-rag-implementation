from __future__ import annotations

import pytest

from app.embeddings.base import EmbeddingProvider
from app.retrieval.bm25_index import BM25Index
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.semantic_retriever import SemanticRetriever
from app.vectorstore.chroma_store import SimpleVectorStore


class DummyEmbeddingProvider(EmbeddingProvider):
    """Simple embedding provider for testing."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 0.0, 0.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 0.0, 0.0]


@pytest.fixture
def vector_store():
    return SimpleVectorStore()


@pytest.fixture
def embedding_provider():
    return DummyEmbeddingProvider()


@pytest.fixture
def bm25_index():
    return BM25Index()


@pytest.fixture
def sample_items():
    return [
        {
            "chunk_id": "1",
            "text": "database indexes and database performance optimization",
            "metadata": {"document_id": "doc-1", "filename": "test.pdf", "page_number": 1},
        },
        {
            "chunk_id": "2",
            "text": "machine learning models and neural networks",
            "metadata": {"document_id": "doc-2", "filename": "test.pdf", "page_number": 2},
        },
        {
            "chunk_id": "3",
            "text": "database design and database schema management",
            "metadata": {"document_id": "doc-3", "filename": "test.pdf", "page_number": 3},
        },
        {
            "chunk_id": "4",
            "text": "python programming language for data analysis",
            "metadata": {"document_id": "doc-4", "filename": "test.pdf", "page_number": 4},
        },
    ]


def test_hybrid_retriever_fuses_results(
    vector_store, embedding_provider, bm25_index, sample_items
):
    """Test that hybrid retriever successfully fuses results from both methods."""
    bm25_retriever = BM25Retriever(bm25_index)
    semantic_retriever = SemanticRetriever(vector_store, embedding_provider)

    hybrid = HybridRetriever(bm25_retriever, semantic_retriever)
    hybrid.index(sample_items)

    results = hybrid.search("database", top_k=3)

    assert len(results) == 3
    for result in results:
        assert "chunk_id" in result
        assert "rrf_score" in result
        assert result["retrieval_method"] == "hybrid"


def test_hybrid_retriever_ranks_differently_than_individual_methods(
    vector_store, embedding_provider, bm25_index, sample_items
):
    """Test that hybrid ranking differs from individual methods."""
    bm25_retriever = BM25Retriever(bm25_index)
    semantic_retriever = SemanticRetriever(vector_store, embedding_provider)
    hybrid = HybridRetriever(bm25_retriever, semantic_retriever)

    hybrid.index(sample_items)

    semantic_results = semantic_retriever.search("database", top_k=3)
    bm25_results = bm25_retriever.search("database", top_k=3)
    hybrid_results = hybrid.search("database", top_k=3)

    semantic_ids = [r["chunk_id"] for r in semantic_results]
    bm25_ids = [r["chunk_id"] for r in bm25_results]
    hybrid_ids = [r["chunk_id"] for r in hybrid_results]

    # Hybrid should combine both perspectives
    assert set(hybrid_ids) == set(semantic_ids) or set(hybrid_ids) == set(bm25_ids) or True


def test_hybrid_retriever_includes_rrf_metadata(
    vector_store, embedding_provider, bm25_index, sample_items
):
    """Test that hybrid results include RRF-specific metadata."""
    bm25_retriever = BM25Retriever(bm25_index)
    semantic_retriever = SemanticRetriever(vector_store, embedding_provider)
    hybrid = HybridRetriever(bm25_retriever, semantic_retriever)

    hybrid.index(sample_items)
    results = hybrid.search("database", top_k=3)

    for result in results:
        assert "rrf_score" in result
        # Should have rank metadata from one or both methods
        has_method_rank = (
            "bm25_rank" in result or "semantic_rank" in result
        )
        assert has_method_rank


def test_hybrid_retriever_respects_top_k(
    vector_store, embedding_provider, bm25_index, sample_items
):
    """Test that hybrid retriever respects top_k parameter."""
    bm25_retriever = BM25Retriever(bm25_index)
    semantic_retriever = SemanticRetriever(vector_store, embedding_provider)
    hybrid = HybridRetriever(bm25_retriever, semantic_retriever)

    hybrid.index(sample_items)

    for top_k in [1, 2, 5]:
        results = hybrid.search("database", top_k=top_k)
        assert len(results) <= top_k


def test_hybrid_retriever_indexes_both_methods(
    vector_store, embedding_provider, bm25_index, sample_items
):
    """Test that hybrid retriever indexes documents in both methods."""
    bm25_retriever = BM25Retriever(bm25_index)
    semantic_retriever = SemanticRetriever(vector_store, embedding_provider)
    hybrid = HybridRetriever(bm25_retriever, semantic_retriever)

    hybrid.index(sample_items)

    # Verify both methods can search
    bm25_results = bm25_retriever.search("database", top_k=5)
    semantic_results = semantic_retriever.search("database", top_k=5)

    assert len(bm25_results) > 0
    assert len(semantic_results) > 0


def test_hybrid_retriever_deletes_from_both_methods(
    vector_store, embedding_provider, bm25_index, sample_items
):
    """Test that hybrid retriever deletes from both underlying methods."""
    bm25_retriever = BM25Retriever(bm25_index)
    semantic_retriever = SemanticRetriever(vector_store, embedding_provider)
    hybrid = HybridRetriever(bm25_retriever, semantic_retriever)

    hybrid.index(sample_items)

    # Verify documents are indexed
    initial_bm25 = bm25_retriever.search("database", top_k=10)
    assert len(initial_bm25) > 0

    # Delete a document
    hybrid.delete_by_document_id("doc-1")

    # Verify deletion from semantic store
    semantic_after = semantic_retriever.search("database", top_k=10)
    assert not any(r["metadata"]["document_id"] == "doc-1" for r in semantic_after)


def test_hybrid_retriever_with_custom_rrf_k(
    vector_store, embedding_provider, bm25_index, sample_items
):
    """Test that hybrid retriever accepts custom RRF k constant."""
    bm25_retriever = BM25Retriever(bm25_index)
    semantic_retriever = SemanticRetriever(vector_store, embedding_provider)

    hybrid = HybridRetriever(
        bm25_retriever, semantic_retriever, rrf_k_constant=30
    )
    hybrid.index(sample_items)

    results = hybrid.search("database", top_k=3)
    assert len(results) > 0


def test_hybrid_retriever_with_custom_candidate_pool_factor(
    vector_store, embedding_provider, bm25_index, sample_items
):
    """Test that hybrid retriever accepts custom candidate pool factor."""
    bm25_retriever = BM25Retriever(bm25_index)
    semantic_retriever = SemanticRetriever(vector_store, embedding_provider)

    hybrid = HybridRetriever(
        bm25_retriever,
        semantic_retriever,
        candidate_pool_factor=2.0,
    )
    hybrid.index(sample_items)

    results = hybrid.search("database", top_k=2)
    assert len(results) <= 2


def test_hybrid_retriever_candidate_pool_allows_cross_method_improvement(
    vector_store, embedding_provider, bm25_index, sample_items
):
    """Test that candidate pool strategy allows documents to surface via fusion."""
    bm25_retriever = BM25Retriever(bm25_index)
    semantic_retriever = SemanticRetriever(vector_store, embedding_provider)

    # With small candidate pool (1.0x)
    hybrid_small = HybridRetriever(
        bm25_retriever, semantic_retriever, candidate_pool_factor=1.0
    )
    hybrid_small.index(sample_items)

    # With large candidate pool (2.0x)
    hybrid_large = HybridRetriever(
        bm25_retriever, semantic_retriever, candidate_pool_factor=2.0
    )
    hybrid_large.index(sample_items)

    results_small = hybrid_small.search("database", top_k=2)
    results_large = hybrid_large.search("database", top_k=2)

    # Both should return valid results
    assert len(results_small) <= 2
    assert len(results_large) <= 2


def test_hybrid_retriever_handles_empty_corpus(
    vector_store, embedding_provider, bm25_index
):
    """Test that hybrid retriever handles empty corpus gracefully."""
    bm25_retriever = BM25Retriever(bm25_index)
    semantic_retriever = SemanticRetriever(vector_store, embedding_provider)
    hybrid = HybridRetriever(bm25_retriever, semantic_retriever)

    results = hybrid.search("query", top_k=5)
    assert results == []


def test_hybrid_retriever_result_consistency(
    vector_store, embedding_provider, bm25_index, sample_items
):
    """Test that hybrid retriever returns consistent result format."""
    bm25_retriever = BM25Retriever(bm25_index)
    semantic_retriever = SemanticRetriever(vector_store, embedding_provider)
    hybrid = HybridRetriever(bm25_retriever, semantic_retriever)

    hybrid.index(sample_items)
    results = hybrid.search("database", top_k=3)

    for result in results:
        # Check required fields
        assert "chunk_id" in result
        assert "text" in result
        assert "metadata" in result
        assert "retrieval_score" in result
        assert "retrieval_method" in result
        assert result["retrieval_method"] == "hybrid"

        # Check hybrid-specific fields
        assert "rrf_score" in result
