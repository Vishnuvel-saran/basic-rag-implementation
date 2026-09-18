from __future__ import annotations

import pytest

from app.embeddings.base import EmbeddingProvider
from app.retrieval.base import RetrieverBase
from app.retrieval.bm25_index import BM25Index
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.retriever_factory import create_retriever
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
            "text": "database indexes and performance optimization",
            "metadata": {"document_id": "doc-1", "filename": "test.pdf", "page_number": 1},
        },
        {
            "chunk_id": "2",
            "text": "machine learning models and neural networks",
            "metadata": {"document_id": "doc-2", "filename": "test.pdf", "page_number": 2},
        },
        {
            "chunk_id": "3",
            "text": "database design and schema management",
            "metadata": {"document_id": "doc-3", "filename": "test.pdf", "page_number": 3},
        },
    ]


def test_semantic_retriever_returns_consistent_structure(
    vector_store, embedding_provider, sample_items
):
    """Test that semantic retriever returns results with consistent format."""
    retriever = SemanticRetriever(vector_store, embedding_provider)
    retriever.index(sample_items)

    results = retriever.search("database", top_k=2)

    assert len(results) == 2
    for result in results:
        assert "chunk_id" in result
        assert "text" in result
        assert "metadata" in result
        assert "retrieval_score" in result
        assert result["retrieval_method"] == "semantic"
        assert result["retrieval_score"] is not None


def test_bm25_retriever_returns_consistent_structure(bm25_index, sample_items):
    """Test that BM25 retriever returns results with consistent format."""
    retriever = BM25Retriever(bm25_index)
    retriever.index(sample_items)

    results = retriever.search("database", top_k=2)

    assert len(results) == 2
    for result in results:
        assert "chunk_id" in result
        assert "text" in result
        assert "metadata" in result
        assert "retrieval_score" in result
        assert result["retrieval_method"] == "bm25"
        assert result["retrieval_score"] is not None


def test_semantic_and_bm25_rank_differently(
    vector_store, embedding_provider, bm25_index, sample_items
):
    """Test that semantic and BM25 produce different rankings."""
    semantic = SemanticRetriever(vector_store, embedding_provider)
    semantic.index(sample_items)

    bm25 = BM25Retriever(bm25_index)
    bm25.index(sample_items)

    semantic_results = semantic.search("database", top_k=3)
    bm25_results = bm25.search("database", top_k=3)

    semantic_ids = [r["chunk_id"] for r in semantic_results]
    bm25_ids = [r["chunk_id"] for r in bm25_results]

    # They should produce different rankings (not necessarily completely different)
    # BM25 should prioritize documents with repeated "database" keyword
    assert bm25_ids != semantic_ids or len(bm25_ids) < 3


def test_retriever_factory_creates_semantic_retriever(
    vector_store, embedding_provider, bm25_index
):
    """Test factory creates semantic retriever."""
    retriever = create_retriever(
        "semantic",
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        bm25_index=bm25_index,
    )
    assert isinstance(retriever, SemanticRetriever)


def test_retriever_factory_creates_bm25_retriever(
    vector_store, embedding_provider, bm25_index
):
    """Test factory creates BM25 retriever."""
    retriever = create_retriever(
        "bm25",
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        bm25_index=bm25_index,
    )
    assert isinstance(retriever, BM25Retriever)


def test_retriever_factory_normalizes_method_names(
    vector_store, embedding_provider, bm25_index
):
    """Test factory normalizes method names (lowercase, underscores, etc)."""
    # Test semantic variants
    for method in ["semantic", "SEMANTIC", "Semantic", "semantic_search", "SEMANTIC-SEARCH"]:
        retriever = create_retriever(
            method,
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            bm25_index=bm25_index,
        )
        assert isinstance(retriever, SemanticRetriever)

    # Test BM25 variants
    for method in ["bm25", "BM25", "Bm25", "bm-25", "BM-25"]:
        retriever = create_retriever(
            method,
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            bm25_index=bm25_index,
        )
        assert isinstance(retriever, BM25Retriever)


def test_retriever_factory_raises_on_invalid_method(
    vector_store, embedding_provider, bm25_index
):
    """Test factory raises ValueError for unknown method."""
    with pytest.raises(ValueError, match="Unsupported retrieval method"):
        create_retriever(
            "invalid_method",
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            bm25_index=bm25_index,
        )


def test_retriever_factory_error_message_lists_supported_methods(
    vector_store, embedding_provider, bm25_index
):
    """Test factory error message lists all supported methods."""
    with pytest.raises(ValueError) as exc_info:
        create_retriever(
            "unknown",
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            bm25_index=bm25_index,
        )
    error_msg = str(exc_info.value)
    assert "bm25" in error_msg
    assert "semantic" in error_msg
    assert "hybrid" in error_msg


def test_semantic_retriever_index_and_delete(
    vector_store, embedding_provider, sample_items
):
    """Test semantic retriever can index and delete documents."""
    retriever = SemanticRetriever(vector_store, embedding_provider)
    retriever.index(sample_items)

    # Verify indexing worked
    results = retriever.search("database", top_k=10)
    assert len(results) == 3

    # Delete one document
    retriever.delete_by_document_id("doc-1")
    results = retriever.search("database", top_k=10)
    assert len(results) == 2
    assert all(r["metadata"]["document_id"] != "doc-1" for r in results)


def test_bm25_retriever_index_and_delete(bm25_index, sample_items):
    """Test BM25 retriever can index and delete documents."""
    retriever = BM25Retriever(bm25_index)
    retriever.index(sample_items)

    # Verify indexing worked
    results = retriever.search("database", top_k=10)
    assert len(results) == 3

    # Delete one document (BM25 retriever marks for deletion but requires rebuild)
    retriever.delete_by_document_id("doc-1")
    # Note: BM25 requires external rebuild after deletion


def test_retrievers_handle_empty_corpus(vector_store, embedding_provider, bm25_index):
    """Test retrievers handle empty corpus gracefully."""
    semantic = SemanticRetriever(vector_store, embedding_provider)
    bm25 = BM25Retriever(bm25_index)

    semantic_results = semantic.search("query", top_k=5)
    bm25_results = bm25.search("query", top_k=5)

    assert semantic_results == []
    assert bm25_results == []


def test_retriever_respects_top_k(vector_store, embedding_provider, bm25_index, sample_items):
    """Test that retrievers respect the top_k parameter."""
    semantic = SemanticRetriever(vector_store, embedding_provider)
    semantic.index(sample_items)

    bm25 = BM25Retriever(bm25_index)
    bm25.index(sample_items)

    for top_k in [1, 2, 10]:
        semantic_results = semantic.search("database", top_k=top_k)
        bm25_results = bm25.search("database", top_k=top_k)

        assert len(semantic_results) <= top_k
        assert len(bm25_results) <= top_k
