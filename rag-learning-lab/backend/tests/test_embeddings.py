from app.embeddings.factory import create_embedding_provider
from app.embeddings.providers.openrouter import OpenRouterEmbeddingProvider


def test_embedding_provider_returns_vectors_for_documents_and_queries():
    provider = create_embedding_provider("local")

    docs = [
        "RAG helps answer questions from documents.",
        "Chunking is used before embedding.",
    ]
    query = "What is chunking?"

    doc_vectors = provider.embed_documents(docs)
    query_vector = provider.embed_query(query)

    assert len(doc_vectors) == 2
    assert len(doc_vectors[0]) == 3
    assert len(query_vector) == 3


def test_openrouter_embedding_provider_preserves_batch_order(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": [
                    {"index": 1, "embedding": [2.0, 2.0]},
                    {"index": 0, "embedding": [1.0, 1.0]},
                ]
            }

    def fake_post(url, headers, json, timeout):
        assert url == OpenRouterEmbeddingProvider.endpoint
        assert json["model"] == "openai/text-embedding-3-small"
        assert json["input"] == ["first", "second"]
        return FakeResponse()

    monkeypatch.setattr("app.embeddings.providers.openrouter.requests.post", fake_post)
    provider = OpenRouterEmbeddingProvider(
        api_key="test-key", model="openai/text-embedding-3-small"
    )

    assert provider.embed_documents(["first", "second"]) == [
        [1.0, 1.0],
        [2.0, 2.0],
    ]
