from app.embeddings.factory import create_embedding_provider
from app.llm.factory import create_llm_provider
from app.rag.pipeline import RAGPipeline
from app.vectorstore.chroma_store import SimpleVectorStore


class FakeLLM:
    def generate(self, prompt: str, **kwargs) -> str:
        return "This is a grounded answer from the supplied context."


def test_rag_pipeline_builds_prompt_and_returns_answer():
    pipeline = RAGPipeline(
        embedding_provider=create_embedding_provider("local"),
        vector_store=SimpleVectorStore(),
        llm_provider=FakeLLM(),
    )

    chunks = [
        {
            "document_id": "doc-1",
            "filename": "sample.pdf",
            "page_number": 1,
            "chunk_id": "chunk-1",
            "chunk_text": "Python is a programming language used for backend services.",
        },
        {
            "document_id": "doc-1",
            "filename": "sample.pdf",
            "page_number": 2,
            "chunk_id": "chunk-2",
            "chunk_text": "FastAPI is a Python web framework.",
        },
    ]

    pipeline.index_chunks(chunks)
    result = pipeline.query("What is FastAPI?", top_k=1)

    assert "What is FastAPI?" in result["prompt"]
    assert "sample.pdf" in result["prompt"]
    assert result["answer"] == "This is a grounded answer from the supplied context."
