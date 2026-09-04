from __future__ import annotations

from app.config.settings import settings
from app.embeddings.base import EmbeddingProvider
from app.llm.base import LLMProvider
from app.rag.prompt_builder import PromptBuilder
from app.vectorstore.chroma_store import SimpleVectorStore


class RAGPipeline:
    """Minimal end-to-end RAG pipeline for learning.

    The design stays deliberately simple:
    - chunk and store documents
    - embed queries and documents
    - retrieve relevant chunks
    - assemble a grounded prompt
    - generate an answer through a provider abstraction
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: SimpleVectorStore,
        llm_provider: LLMProvider,
        top_k: int | None = None,
    ):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.llm_provider = llm_provider
        self.top_k = top_k if top_k is not None else settings.top_k

    def index_chunks(self, chunks: list[dict]) -> list[dict]:
        if not chunks:
            return []

        texts = [chunk.get("chunk_text", "") for chunk in chunks]
        document_vectors = self.embedding_provider.embed_documents(texts)

        stored_items: list[dict] = []
        for chunk, vector in zip(chunks, document_vectors):
            metadata = {
                "document_id": chunk.get("document_id", "unknown"),
                "filename": chunk.get("filename", "unknown"),
                "page_number": chunk.get("page_number", 0),
                "chunk_id": chunk.get("chunk_id", "unknown"),
            }
            self.vector_store.add(
                chunk_id=str(metadata["chunk_id"]),
                text=str(chunk.get("chunk_text", "")),
                vector=vector,
                metadata=metadata,
            )
            stored_items.append({**chunk, "metadata": metadata})

        return stored_items

    def query(self, question: str, top_k: int | None = None) -> dict:
        effective_top_k = self.top_k if top_k is None else top_k
        query_vector = self.embedding_provider.embed_query(question)
        retrieved_chunks = self.vector_store.query(
            query_vector=query_vector, top_k=effective_top_k
        )

        prompt = PromptBuilder.build(question, retrieved_chunks, top_k=effective_top_k)
        answer = self.llm_provider.generate(prompt)

        return {
            "question": question,
            "top_k": effective_top_k,
            "retrieved_chunks": retrieved_chunks,
            "answer": answer,
            "prompt": prompt,
        }
