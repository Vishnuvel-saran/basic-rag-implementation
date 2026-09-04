from __future__ import annotations

from typing import Any


class SimpleVectorStore:
    """A tiny in-memory vector store used to teach the retrieval concept.

    In a real project, this would be backed by ChromaDB or another vector database.
    This class demonstrates the important idea: storing text + vector + metadata and
    then retrieving the nearest chunks by similarity.
    """

    def __init__(self):
        self._items: list[dict[str, Any]] = []

    def add(
        self, chunk_id: str, text: str, vector: list[float], metadata: dict[str, Any]
    ) -> None:
        self._items.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "vector": vector,
                "metadata": metadata,
            }
        )

    def query(self, query_vector: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        scored: list[tuple[float, dict[str, Any]]] = []

        for item in self._items:
            similarity = self._cosine_similarity(query_vector, item["vector"])
            scored.append((similarity, item))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if len(a) != len(b):
            raise ValueError(
                "Vectors must be the same length to compare cosine similarity."
            )

        dot = sum(x * y for x, y in zip(a, b))
        mag_a = sum(x * x for x in a) ** 0.5
        mag_b = sum(x * x for x in b) ** 0.5

        if mag_a == 0 or mag_b == 0:
            return 0.0

        return dot / (mag_a * mag_b)
