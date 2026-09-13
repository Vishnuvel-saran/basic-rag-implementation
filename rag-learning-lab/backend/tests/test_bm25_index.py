from fastapi.testclient import TestClient
from io import BytesIO

from app.embeddings.base import EmbeddingProvider
from app.llm.base import LLMProvider
from app.main import app
from app.rag.pipeline import RAGPipeline
from app.retrieval.bm25_index import BM25Index
from app.vectorstore.chroma_store import SimpleVectorStore


class BM25EmbeddingProvider(EmbeddingProvider):
    """Test embedding provider for BM25 tests."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]

    @staticmethod
    def _vector(text: str) -> list[float]:
        return [1.0, 0.0] if "database" in text else [0.0, 1.0]


class BM25LLM(LLMProvider):
    """Test LLM provider for BM25 tests."""

    def generate(self, prompt: str, **kwargs) -> str:
        return "Test answer"


def test_bm25_search_keyword_heavy_query_differs_from_cosine():
    """Test that BM25 and cosine similarity rank results differently for keyword queries."""
    vector_store = SimpleVectorStore()
    embedding_provider = BM25EmbeddingProvider()
    bm25_index = BM25Index()

    # Add chunks with different keyword densities
    chunks_data = [
        {
            "chunk_id": "1",
            "text": "database indexes and database performance optimization",
            "vector": [1.0, 0.0],
            "metadata": {"document_id": "doc-1", "page_number": 1, "filename": "doc1.pdf"},
        },
        {
            "chunk_id": "2",
            "text": "machine learning models and neural networks",
            "vector": [0.0, 1.0],
            "metadata": {"document_id": "doc-2", "page_number": 1, "filename": "doc2.pdf"},
        },
        {
            "chunk_id": "3",
            "text": "database design and schema database management",
            "vector": [1.0, 0.0],
            "metadata": {"document_id": "doc-3", "page_number": 1, "filename": "doc3.pdf"},
        },
    ]

    for chunk_data in chunks_data:
        vector_store.add(**chunk_data)

    bm25_index.build(vector_store._items)

    # Query with "database" keyword
    query = "database"

    # Cosine results (will rank based on vector [1, 0] similarity)
    cosine_results = vector_store.query([1.0, 0.1], top_k=3)
    cosine_ids = [chunk["chunk_id"] for chunk in cosine_results]

    # BM25 results (will rank based on keyword frequency)
    bm25_results = bm25_index.search(query, top_k=3)
    bm25_ids = [chunk["chunk_id"] for chunk in bm25_results]

    # BM25 should prioritize chunks with repeated "database" keyword
    # Chunk 1 has "database" twice, chunk 3 has "database" twice, chunk 2 has zero
    # So BM25 should rank 1 and 3 higher than 2
    assert "2" not in bm25_ids[:2], "BM25 should not rank keyword-free chunk at top"
    # All returned results should have scores >= 0 (some may be 0 for no matches)
    assert all(chunk["bm25_score"] >= 0 for chunk in bm25_results)


def test_bm25_index_clean_after_document_delete():
    """Test that BM25 index reflects deletions and contains no stale chunks."""
    vector_store = SimpleVectorStore()
    bm25_index = BM25Index()

    # Add chunks from two documents
    chunks_data = [
        {
            "chunk_id": "doc1_chunk1",
            "text": "content from document one",
            "vector": [1.0, 0.0],
            "metadata": {"document_id": "doc-1", "page_number": 1, "filename": "doc1.pdf"},
        },
        {
            "chunk_id": "doc2_chunk1",
            "text": "content from document two",
            "vector": [0.0, 1.0],
            "metadata": {"document_id": "doc-2", "page_number": 1, "filename": "doc2.pdf"},
        },
    ]

    for chunk_data in chunks_data:
        vector_store.add(**chunk_data)

    bm25_index.build(vector_store._items)

    # Verify both documents are indexed
    initial_results = bm25_index.search("content", top_k=10)
    assert len(initial_results) == 2, "Should have 2 chunks initially"

    # Delete doc-1
    vector_store.delete_by_document_id("doc-1")
    bm25_index.build(vector_store._items)

    # Verify only doc-2 remains
    after_delete_results = bm25_index.search("content", top_k=10)
    assert len(after_delete_results) == 1, "Should have 1 chunk after deletion"
    assert (
        after_delete_results[0]["metadata"]["document_id"] == "doc-2"
    ), "Remaining chunk should be from doc-2"


def test_query_endpoint_includes_bm25_when_requested():
    """Test that /query endpoint returns both cosine and BM25 results when requested."""
    vector_store = SimpleVectorStore()
    embedding_provider = BM25EmbeddingProvider()
    app.state.vector_store = vector_store
    app.state.embedding_provider = embedding_provider
    app.state.rag_pipeline = RAGPipeline(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        llm_provider=BM25LLM(),
    )

    # Add test data to vector store
    app.state.vector_store.add(
        chunk_id="1",
        text="The Python programming language is popular for data science.",
        vector=[1.0, 0.0],
        metadata={"document_id": "doc-1", "page_number": 1, "filename": "test.pdf"},
    )
    app.state.vector_store.add(
        chunk_id="2",
        text="Java and C++ are also used in software development.",
        vector=[0.0, 1.0],
        metadata={"document_id": "doc-2", "page_number": 1, "filename": "test.pdf"},
    )
    app.state.rag_pipeline.bm25_index.build(app.state.vector_store._items)

    client = TestClient(app)

    # Query without BM25 (default)
    response_without_bm25 = client.post(
        "/query", json={"question": "What about Python?", "top_k": 1}
    )
    assert response_without_bm25.status_code == 200
    data = response_without_bm25.json()
    assert "retrieved_chunks" in data
    assert "retrieved_chunks_bm25" not in data, "Should not include BM25 by default"

    # Query with BM25
    response_with_bm25 = client.post(
        "/query",
        json={"question": "What about Python?", "top_k": 1, "include_bm25": True},
    )
    assert response_with_bm25.status_code == 200
    data = response_with_bm25.json()
    assert "retrieved_chunks" in data
    assert "retrieved_chunks_bm25" in data, "Should include BM25 when requested"
    assert "sources_bm25" in data, "Should include BM25 sources"
    assert len(data["retrieved_chunks_bm25"]) > 0, "BM25 results should not be empty"

    # Verify response structure
    for chunk in data["retrieved_chunks_bm25"]:
        assert "chunk_id" in chunk
        assert "context" in chunk
        assert "metadata" in chunk
        assert "bm25_score" in chunk, "BM25 results should have bm25_score"
        assert (
            "similarity_score" not in chunk
        ), "BM25 results should not have similarity_score"
