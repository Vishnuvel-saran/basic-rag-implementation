from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract interface for all LLM providers.

    The RAG pipeline depends on this interface, not on a specific API or vendor.
    """

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError
