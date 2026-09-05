from __future__ import annotations

import os

from app.config.settings import settings
from app.embeddings.base import EmbeddingProvider
from app.embeddings.providers.openrouter import OpenRouterEmbeddingProvider


class LocalDummyEmbeddingProvider(EmbeddingProvider):
    """A simple local embedding stub for learning purposes.

    This is intentionally not a production embedding provider. It demonstrates the
    interface shape: documents and queries are converted into vectors, even though
    the actual vector values are generated in a very simple way for teaching.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._simple_vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._simple_vector(text)

    def _simple_vector(self, text: str) -> list[float]:
        cleaned = text.lower().strip()
        if not cleaned:
            return [0.0, 0.0, 0.0]

        tokens = cleaned.split()
        vector = [0.0, 0.0, 0.0]
        for index, token in enumerate(tokens[:3]):
            value = sum(ord(char) for char in token) / 100.0
            vector[index] = round(value, 4)
        return vector


def create_embedding_provider(provider_name: str | None = None) -> EmbeddingProvider:
    provider = (provider_name or os.getenv("EMBEDDING_PROVIDER", "local")).lower()
    if provider in {"local", "dummy"}:
        return LocalDummyEmbeddingProvider()
    if provider in {"openrouter", "open_router"}:
        return OpenRouterEmbeddingProvider(
            api_key=settings.embedding_api_key,
            model=settings.embedding_model,
        )
    raise ValueError(f"Unsupported embedding provider: {provider}")
