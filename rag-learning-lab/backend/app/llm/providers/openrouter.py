from __future__ import annotations

import os

import requests

from app.llm.base import LLMProvider


class OpenRouterProvider(LLMProvider):
    """OpenRouter-backed provider.

    This is intentionally thin and normalized: it accepts a prompt string and
    sends a standard OpenAI-compatible chat request to the OpenRouter API.
    """

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.model = model or os.getenv("LLM_MODEL", "openai/gpt-4o-mini")

    def build_payload(self, prompt: str, **kwargs) -> dict:
        return {
            "model": kwargs.get("model") or self.model,
            "messages": [{"role": "user", "content": prompt}],
        }

    def generate(self, prompt: str, **kwargs) -> str:
        return self.generate_with_metadata(prompt, **kwargs)["answer"]

    def generate_with_metadata(self, prompt: str, **kwargs) -> dict:
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is required to use the OpenRouter provider."
            )

        payload = self.build_payload(prompt, model=kwargs.get("model"))
        if kwargs.get("temperature") is not None:
            payload["temperature"] = kwargs["temperature"]
        if kwargs.get("max_output_tokens") is not None:
            payload["max_tokens"] = kwargs["max_output_tokens"]
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": kwargs.get("http_referer", "http://localhost:8000"),
            "X-Title": kwargs.get("app_name", "rag-learning-lab"),
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise ValueError("OpenRouter returned no message choices.")

        message = choices[0].get("message", {})
        return {
            "answer": message.get("content", ""),
            "usage": data.get("usage"),
            "model": data.get("model", self.model),
        }
