from __future__ import annotations

from app.vectorstore.chroma_store import SimpleVectorStore


class Retriever:
    def __init__(self, vector_store: SimpleVectorStore):
        self.vector_store = vector_store

    def retrieve(self, query_vector: list[float], top_k: int) -> list[dict]:
        return self.vector_store.query(query_vector=query_vector, top_k=top_k)
