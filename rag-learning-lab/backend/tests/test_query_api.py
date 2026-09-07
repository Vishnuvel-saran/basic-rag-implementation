from fastapi.testclient import TestClient
from io import BytesIO

from app.embeddings.factory import create_embedding_provider
from app.main import app
from app.vectorstore.chroma_store import SimpleVectorStore


class FakeLLM:
    def generate(self, prompt: str, **kwargs) -> str:
        return "This is a grounded answer from the supplied context."


def test_query_endpoint_returns_answer_for_indexed_document():
    app.state.vector_store = SimpleVectorStore()
    app.state.rag_pipeline = type(
        "DummyPipeline",
        (),
        {
            "query": lambda self, question, top_k=5: {
                "answer": "This is a grounded answer from the supplied context.",
                "retrieved_chunks": [
                    {
                        "chunk_id": "2_0",
                        "text": "FastAPI is a Python web framework.",
                        "vector": [0.1, 0.2, 0.3],
                        "metadata": {"filename": "sample.pdf", "page_number": 2},
                    }
                ],
                "question": question,
                "top_k": top_k,
                "prompt": "prompt text",
            }
        },
    )()

    client = TestClient(app)
    response = client.post("/query", json={"question": "What is FastAPI?", "top_k": 1})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "This is a grounded answer from the supplied context."
    assert data["question"] == "What is FastAPI?"
    assert "sample.pdf" in data["sources"][0]
    assert data["retrieved_chunks"] == [
        {
            "chunk_id": "2_0",
            "context": "FastAPI is a Python web framework.",
            "metadata": {"filename": "sample.pdf", "page_number": 2},
        }
    ]
    assert "vector" not in data["retrieved_chunks"][0]


def test_upload_endpoint_passes_chunking_strategy(monkeypatch):
    captured = {}

    def fake_chunk_document_pages(pages, **kwargs):
        captured.update(kwargs)
        return [
            {
                "document_id": "demo.pdf",
                "filename": "demo.pdf",
                "page_number": 1,
                "chunk_id": "chunk_001",
                "chunk_text": "A sentence.",
                "chunking_strategy": kwargs["strategy"],
            }
        ]

    class DummyPipeline:
        def index_chunks(self, chunks):
            return [{**chunks[0], "metadata": {"chunk_id": "chunk_001"}}]

    monkeypatch.setattr("app.main.save_uploaded_pdf", lambda file, filename: "demo.pdf")
    monkeypatch.setattr(
        "app.main.extract_document_from_pdf",
        lambda path: {
            "filename": "demo.pdf",
            "page_count": 1,
            "pages": [{"page_number": 1, "text": "A sentence."}],
        },
    )
    monkeypatch.setattr("app.main.chunk_document_pages", fake_chunk_document_pages)
    app.state.rag_pipeline = DummyPipeline()

    client = TestClient(app)
    response = client.post(
        "/documents/upload?chunking_strategy=paragraph&chunk_size=200&chunk_overlap=0",
        files={"file": ("demo.pdf", BytesIO(b"%PDF-1.4 demo"), "application/pdf")},
    )

    assert response.status_code == 200
    assert captured["strategy"] == "paragraph"
    assert response.json()["chunking_strategy"] == "paragraph"
