from app.ingestion.chunker import chunk_document_pages
from app.ingestion.chunker_factory import ChunkerFactory
from app.ingestion.chunkers import ChunkingConfig, RecursiveChunker


class FakeAgenticLLM:
    def __init__(
        self, response: str = '{"boundaries":[2]}', error: Exception | None = None
    ):
        self.response = response
        self.error = error
        self.prompts: list[str] = []

    def generate(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return self.response


def test_agentic_chunker_uses_llm_boundaries_with_exact_offsets():
    text = "Alpha one. Beta two. Gamma three."
    provider = FakeAgenticLLM('{"boundaries":[2]}')
    chunker = ChunkerFactory.create(
        "agentic",
        ChunkingConfig(chunk_size=20, chunk_overlap=0),
        llm_provider=provider,
    )

    spans = chunker.chunk_with_offsets(text)
    words = text.split()

    assert [(span.start, span.end) for span in spans] == [(0, 4), (4, 6)]
    assert [span.text for span in spans] == [
        " ".join(words[span.start : span.end]) for span in spans
    ]
    assert provider.prompts
    assert "Do not rewrite" in provider.prompts[0]


def test_agentic_chunker_handles_empty_input_without_calling_llm():
    provider = FakeAgenticLLM()
    chunker = ChunkerFactory.create(
        "agentic", ChunkingConfig(chunk_size=20, chunk_overlap=0), llm_provider=provider
    )

    assert chunker.chunk_with_offsets("") == []
    assert provider.prompts == []


def test_agentic_chunker_falls_back_to_recursive_on_llm_failure():
    text = "Alpha one. Beta two. Gamma three. Delta four."
    config = ChunkingConfig(chunk_size=3, chunk_overlap=0)
    provider = FakeAgenticLLM(error=TimeoutError("LLM timed out"))
    chunker = ChunkerFactory.create("agentic", config, llm_provider=provider)

    actual = chunker.chunk_with_offsets(text)
    expected = RecursiveChunker(config).chunk_with_offsets(text)

    assert [span.text for span in actual] == [span.text for span in expected]
    assert [(span.start, span.end) for span in actual] == [
        (span.start, span.end) for span in expected
    ]


def test_agentic_chunker_preserves_repeated_boilerplate_page_attribution():
    provider = FakeAgenticLLM('{"boundaries":[1,2,3]}')
    pages = [
        {
            "page_number": 1,
            "document_id": "doc-1",
            "filename": "doc.pdf",
            "text": "Repeated notice. Alpha content.",
        },
        {
            "page_number": 2,
            "document_id": "doc-1",
            "filename": "doc.pdf",
            "text": "Repeated notice. Beta content.",
        },
    ]

    chunks = chunk_document_pages(
        pages,
        chunk_size=20,
        chunk_overlap=0,
        strategy="agentic",
        llm_provider=provider,
    )

    repeated_notice_chunks = [
        chunk for chunk in chunks if chunk["chunk_text"] == "Repeated notice."
    ]
    assert [chunk["page_ids"] for chunk in repeated_notice_chunks] == [[1], [2]]
    assert next(
        chunk["page_ids"] for chunk in chunks if "Alpha" in chunk["chunk_text"]
    ) == [1]
    assert next(
        chunk["page_ids"] for chunk in chunks if "Beta" in chunk["chunk_text"]
    ) == [2]
