from __future__ import annotations

import os

import requests

from app.embeddings.base import EmbeddingProvider


class OpenRouterEmbeddingProvider(EmbeddingProvider):
    """Embedding provider backed by OpenRouter's OpenAI-compatible API."""

    endpoint = "https://openrouter.ai/api/v1/embeddings"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or os.getenv(
            "EMBEDDING_API_KEY", os.getenv("OPENROUTER_API_KEY", "")
        )
        self.model = model or os.getenv(
            "EMBEDDING_MODEL", "openai/text-embedding-3-small"
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        embeddings = self._embed([text])
        return embeddings[0]

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is required to use the OpenRouter embedding provider."
            )

        try:
            response = requests.post(
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost:8000",
                    "X-OpenRouter-Title": "rag-learning-lab",
                },
                json={
                    "model": self.model,
                    "input": inputs,
                    "encoding_format": "float",
                },
                timeout=60,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            response_text = getattr(exc.response, "text", "")[:500]
            raise RuntimeError(
                f"OpenRouter embedding request failed: {response_text or exc}"
            ) from exc

        data = response.json()
        items = data.get("data", [])
        if len(items) != len(inputs):
            raise ValueError("OpenRouter returned an unexpected number of embeddings.")

        ordered_items = sorted(items, key=lambda item: item.get("index", 0))
        embeddings = [item.get("embedding") for item in ordered_items]
        if any(not isinstance(embedding, list) for embedding in embeddings):
            raise ValueError("OpenRouter returned an invalid embedding response.")

        return embeddings
