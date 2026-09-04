from app.llm.factory import create_llm_provider


def test_openrouter_factory_creates_provider():
    provider = create_llm_provider("openrouter")
    assert provider.__class__.__name__ == "OpenRouterProvider"


def test_openrouter_provider_builds_chat_payload():
    provider = create_llm_provider("openrouter")
    payload = provider.build_payload("Explain embeddings.")

    assert payload["model"]
    assert payload["messages"][0]["role"] == "user"
    assert "Explain embeddings." in payload["messages"][0]["content"]
