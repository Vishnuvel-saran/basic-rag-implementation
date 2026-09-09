from __future__ import annotations

from app.embeddings.base import EmbeddingProvider
from app.ingestion.chunkers import (
    Chunker,
    ChunkingConfig,
    FixedSizeChunker,
    AgenticChunker,
    ParagraphChunker,
    RecursiveChunker,
    SemanticChunker,
    SentenceChunker,
)


class ChunkerFactory:
    """Create one chunking strategy without coupling ingestion to its implementation."""

    @staticmethod
    def create(
        strategy: str,
        config: ChunkingConfig,
        embedding_provider: EmbeddingProvider | None = None,
        llm_provider=None,
    ) -> Chunker:
        normalized = strategy.strip().lower().replace("_", "-")
        chunkers = {
            "fixed": FixedSizeChunker,
            "fixed-size": FixedSizeChunker,
            "sentence": SentenceChunker,
            "sentence-based": SentenceChunker,
            "paragraph": ParagraphChunker,
            "paragraph-based": ParagraphChunker,
            "recursive": RecursiveChunker,
        }
        chunker_type = chunkers.get(normalized)
        if chunker_type:
            uses_overlap = normalized in {
                "fixed",
                "fixed-size",
                "paragraph",
                "paragraph-based",
                "recursive",
            }
            config.validate(uses_overlap=uses_overlap)
            return chunker_type(config)
        if normalized in {"semantic", "semantic-based"}:
            config.validate(uses_overlap=False)
            return SemanticChunker(config, embedding_provider)
        if normalized == "agentic":
            config.validate(uses_overlap=True)
            return AgenticChunker(config, llm_provider)
        raise ValueError(
            "Unsupported chunking strategy. Choose fixed, sentence, paragraph, recursive, semantic, or agentic."
        )
