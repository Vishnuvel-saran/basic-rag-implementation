from app.embeddings.factory import create_embedding_provider


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
