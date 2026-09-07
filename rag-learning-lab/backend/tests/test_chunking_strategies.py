from app.embeddings.base import EmbeddingProvider
from app.ingestion.chunker import chunk_document_pages
from app.ingestion.chunker_factory import ChunkerFactory
from app.ingestion.chunkers import ChunkingConfig


class FakeSemanticEmbeddingProvider(EmbeddingProvider):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vectors.append([1.0, 0.0] if "fruit" in text.lower() else [0.0, 1.0])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


def test_all_non_semantic_strategies_handle_empty_text():
    config = ChunkingConfig(chunk_size=5, chunk_overlap=1)
    for strategy in ["fixed", "sentence", "paragraph", "recursive"]:
        assert ChunkerFactory.create(strategy, config).chunk("") == []


def test_sentence_chunker_keeps_sentence_boundaries():
    chunker = ChunkerFactory.create(
        "sentence", ChunkingConfig(chunk_size=8, chunk_overlap=50)
    )

    chunks = chunker.chunk(
        "First sentence is here. Second sentence follows. Third topic starts now."
    )

    assert len(chunks) == 2
    assert all(chunk.endswith(".") for chunk in chunks)
    assert "First sentence is here." in chunks[0]
    assert "Second sentence follows." in chunks[0]


def test_sentence_chunker_does_not_validate_unused_overlap():
    chunker = ChunkerFactory.create(
        "sentence", ChunkingConfig(chunk_size=10, chunk_overlap=50)
    )

    assert chunker.chunk("One sentence. Another sentence.")


def test_paragraph_chunker_preserves_paragraph_structure():
    chunker = ChunkerFactory.create(
        "paragraph", ChunkingConfig(chunk_size=20, chunk_overlap=0)
    )

    chunks = chunker.chunk("Heading\n\nFirst paragraph.\n\nSecond paragraph.")

    assert len(chunks) == 1
    assert "First paragraph." in chunks[0]
    assert "Second paragraph." in chunks[0]


def test_recursive_chunker_falls_back_for_large_text():
    chunker = ChunkerFactory.create(
        "recursive", ChunkingConfig(chunk_size=4, chunk_overlap=1)
    )

    chunks = chunker.chunk("One two three four five six seven eight nine ten.")

    assert len(chunks) > 1
    assert all(chunk for chunk in chunks)
    assert chunks[1].split()[0] in chunks[0].split()


def test_semantic_chunker_splits_when_adjacent_topic_changes():
    chunker = ChunkerFactory.create(
        "semantic",
        ChunkingConfig(chunk_size=50, chunk_overlap=0, semantic_threshold=0.8),
        embedding_provider=FakeSemanticEmbeddingProvider(),
    )

    chunks = chunker.chunk(
        "Apple is a fruit. Banana is a fruit. Servers use memory. Databases store records."
    )

    assert len(chunks) == 2
    assert "Apple" in chunks[0]
    assert "Servers" in chunks[1]


def test_page_chunk_metadata_records_strategy_and_index():
    chunks = chunk_document_pages(
        [
            {
                "page_number": 3,
                "document_id": "doc-1",
                "filename": "x.pdf",
                "text": "One. Two.",
            }
        ],
        chunk_size=10,
        chunk_overlap=0,
        strategy="sentence",
    )

    assert chunks[0]["chunk_id"] == "chunk_001"
    assert chunks[0]["chunking_strategy"] == "sentence"
    assert chunks[0]["chunk_index"] == 0
