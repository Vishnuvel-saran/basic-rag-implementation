from __future__ import annotations

import os

from app.llm.base import LLMProvider
from app.llm.providers.openrouter import OpenRouterProvider


def create_llm_provider(provider_name: str | None = None) -> LLMProvider:
    provider = (provider_name or os.getenv("LLM_PROVIDER", "openrouter")).lower()

    if provider in {"openrouter", "open_router"}:
        return OpenRouterProvider(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model=os.getenv("LLM_MODEL", "openai/gpt-4o-mini"),
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")
