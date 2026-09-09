from io import BytesIO

from fastapi.testclient import TestClient

from app.embeddings.base import EmbeddingProvider
from app.llm.base import LLMProvider
from app.main import app
from app.rag.pipeline import RAGPipeline
from app.vectorstore.chroma_store import SimpleVectorStore


class DocumentEmbeddingProvider(EmbeddingProvider):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]

    @staticmethod
    def _vector(text: str) -> list[float]:
        return [1.0, 0.0] if "doc-1" in text else [0.0, 1.0]


class DocumentLLM(LLMProvider):
    def generate(self, prompt: str, **kwargs) -> str:
        return "document answer"


def test_document_cap_replacement_listing_deletion_and_global_query(monkeypatch):
    vector_store = SimpleVectorStore()
    embedding_provider = DocumentEmbeddingProvider()
    app.state.vector_store = vector_store
    app.state.embedding_provider = embedding_provider
    app.state.rag_pipeline = RAGPipeline(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        llm_provider=DocumentLLM(),
    )

    extracted_calls = []

    def fake_save(file_obj, filename):
        return filename

    def fake_extract(path):
        filename = str(path)
        extracted_calls.append(filename)
        return {
            "filename": filename,
            "page_count": 2,
            "pages": [
                {"page_number": 1, "text": f"content for {filename}"},
                {"page_number": 2, "text": f"more content for {filename}"},
            ],
        }

    def fake_chunk(pages, **kwargs):
        document_id = pages[0]["document_id"]
        return [
            {
                "document_id": document_id,
                "filename": document_id,
                "page_number": 1,
                "page_ids": [1, 2],
                "chunk_id": "chunk_001",
                "chunk_text": f"content for {document_id} doc-1"
                if document_id == "doc-1.pdf"
                else f"content for {document_id}",
                "chunking_strategy": "fixed",
                "chunk_index": 0,
                "chunk_size": 400,
            }
        ]

    monkeypatch.setattr("app.main.save_uploaded_pdf", fake_save)
    monkeypatch.setattr("app.main.extract_document_from_pdf", fake_extract)
    monkeypatch.setattr("app.main.chunk_document_pages", fake_chunk)

    client = TestClient(app)

    for index in range(1, 5):
        response = client.post(
            "/documents/upload",
            files={"file": (f"doc-{index}.pdf", BytesIO(b"pdf"), "application/pdf")},
        )
        assert response.status_code == 200

    listed = client.get("/documents")
    assert listed.status_code == 200
    assert listed.json()["count"] == 4
    assert listed.json()["max_documents"] == 4

    extracted_calls.clear()
    fifth = client.post(
        "/documents/upload",
        files={"file": ("doc-5.pdf", BytesIO(b"pdf"), "application/pdf")},
    )
    assert fifth.status_code == 400
    assert "Maximum of 4 documents allowed" in fifth.json()["detail"]
    assert extracted_calls == []
    assert client.get("/documents").json()["count"] == 4

    replacement = client.post(
        "/documents/upload",
        files={"file": ("doc-1.pdf", BytesIO(b"new pdf"), "application/pdf")},
    )
    assert replacement.status_code == 200
    assert client.get("/documents").json()["count"] == 4
    assert len(vector_store._items) == 4

    removed = client.delete("/documents/doc-2.pdf")
    assert removed.status_code == 200
    assert removed.json()["remaining_count"] == 3
    assert client.get("/documents").json()["count"] == 3

    query = client.post(
        "/query",
        json={"question": "What is in doc-1?", "top_k": 10},
    )
    assert query.status_code == 200
    assert all("doc-2.pdf" not in source for source in query.json()["sources"])
    assert all(
        chunk["metadata"]["filename"] != "doc-2.pdf"
        for chunk in query.json()["retrieved_chunks"]
    )
