from app.vectorstore.chroma_store import ChromaVectorStore


def test_chroma_vector_store_returns_top_matches_for_query():
    store = ChromaVectorStore(
        collection_name="test_collection", persist_directory="./data/chroma_test"
    )

    store.add(
        chunk_id="chunk-1",
        text="The project uses Python and FastAPI for the backend.",
        vector=[1.0, 0.0, 0.0],
        metadata={"document_id": "doc-1", "page_number": 1},
    )
    store.add(
        chunk_id="chunk-2",
        text="The frontend uses React for the user interface.",
        vector=[0.0, 1.0, 0.0],
        metadata={"document_id": "doc-2", "page_number": 2},
    )

    matches = store.query(query_vector=[1.0, 0.1, 0.0], top_k=1)

    assert matches[0]["chunk_id"] == "chunk-1"
    assert matches[0]["metadata"]["document_id"] == "doc-1"
