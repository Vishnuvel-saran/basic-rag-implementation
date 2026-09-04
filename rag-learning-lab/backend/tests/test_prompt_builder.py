from app.rag.prompt_builder import PromptBuilder


def test_prompt_builder_includes_question_and_context_sources():
    retrieved = [
        {
            "text": "Python is a general-purpose programming language.",
            "metadata": {"filename": "notes.pdf", "page_number": 2},
        }
    ]

    prompt = PromptBuilder.build("What is Python?", retrieved, top_k=1)

    assert "What is Python?" in prompt
    assert "notes.pdf" in prompt
    assert "page 2" in prompt
    assert "Answer using the supplied context only." in prompt
