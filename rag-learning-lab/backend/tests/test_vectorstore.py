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


def test_chroma_vector_store_can_replace_a_document_without_duplicates():
    store = ChromaVectorStore()
    store.add(
        chunk_id="chunk-1",
        text="old text",
        vector=[1.0, 0.0],
        metadata={"document_id": "doc-1"},
    )
    store.add(
        chunk_id="chunk-2",
        text="other document",
        vector=[0.9, 0.1],
        metadata={"document_id": "doc-2"},
    )

    store.delete_by_document_id("doc-1")
    store.add(
        chunk_id="chunk-1",
        text="new text",
        vector=[1.0, 0.0],
        metadata={"document_id": "doc-1"},
    )

    matches = store.query(query_vector=[1.0, 0.0], top_k=10)

    assert [item["chunk_id"] for item in matches] == ["chunk-1", "chunk-2"]
    assert matches[0]["text"] == "new text"
