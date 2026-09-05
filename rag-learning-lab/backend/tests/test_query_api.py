from fastapi.testclient import TestClient

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
                        "text": "FastAPI is a Python web framework.",
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
