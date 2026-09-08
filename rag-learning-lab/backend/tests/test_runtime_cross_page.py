from io import BytesIO

import fitz
from fastapi.testclient import TestClient

from app.embeddings.base import EmbeddingProvider
from app.llm.base import LLMProvider
from app.main import app
from app.vectorstore.chroma_store import SimpleVectorStore
from app.rag.pipeline import RAGPipeline


class ControlledEmbeddingProvider(EmbeddingProvider):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.lower()
        return [
            float(
                any(
                    term in lowered
                    for term in ["aurora", "caching", "latency", "redis"]
                )
            ),
            float("unrelated" in lowered),
        ]


class ControlledLLM(LLMProvider):
    def generate(self, prompt: str, **kwargs) -> str:
        return "controlled answer"


def _two_page_pdf() -> BytesIO:
    document = fitz.open()
    first_page = document.new_page()
    first_page.insert_text(
        (50, 80),
        "The Aurora project introduced a distributed caching architecture. "
        "The primary goal was to reduce database latency.",
    )
    second_page = document.new_page()
    second_page.insert_text(
        (50, 80),
        "This architecture uses Redis as the caching layer. "
        "Redis stores frequently accessed objects before requests reach PostgreSQL.",
    )
    buffer = BytesIO(document.tobytes())
    document.close()
    buffer.seek(0)
    return buffer


def test_real_pdf_upload_query_preserves_cross_page_semantic_chunk(monkeypatch):
    embedding_provider = ControlledEmbeddingProvider()
    app.state.vector_store = SimpleVectorStore()
    app.state.embedding_provider = embedding_provider
    app.state.rag_pipeline = RAGPipeline(
        embedding_provider=embedding_provider,
        vector_store=app.state.vector_store,
        llm_provider=ControlledLLM(),
    )

    client = TestClient(app)
    upload = client.post(
        "/documents/upload?chunking_strategy=semantic&chunk_size=80&semantic_threshold=0.4",
        files={
            "file": (
                "controlled-cross-page.pdf",
                _two_page_pdf(),
                "application/pdf",
            )
        },
    )

    assert upload.status_code == 200
    chunks = upload.json()["chunks"]
    assert len(chunks) == 1
    assert chunks[0]["metadata"]["page_ids"] == [1, 2]

    query = client.post(
        "/query",
        json={
            "question": "How does Aurora reduce latency and what technology is used?",
            "top_k": 1,
        },
    )

    assert query.status_code == 200
    result = query.json()
    assert result["retrieved_chunks"][0]["metadata"]["page_ids"] == [1, 2]
    assert "page 1, 2" in result["prompt"]
    assert "pages 1, 2" in result["sources"][0]
