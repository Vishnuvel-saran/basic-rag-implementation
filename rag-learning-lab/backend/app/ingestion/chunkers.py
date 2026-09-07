from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.embeddings.base import EmbeddingProvider


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int = 800
    chunk_overlap: int = 100
    semantic_threshold: float = 0.75

    def validate(self, *, uses_overlap: bool = True) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if uses_overlap:
            if self.chunk_overlap < 0:
                raise ValueError("chunk_overlap must be greater than or equal to 0")
            if self.chunk_overlap >= self.chunk_size:
                raise ValueError("chunk_overlap must be smaller than chunk_size")
        if not 0 <= self.semantic_threshold <= 1:
            raise ValueError("semantic_threshold must be between 0 and 1")


class Chunker(ABC):
    """Strategy boundary: text in, ordered text chunks out."""

    def __init__(self, config: ChunkingConfig):
        self.config = config

    @abstractmethod
    def chunk(self, text: str) -> list[str]:
        raise NotImplementedError


class FixedSizeChunker(Chunker):
    """Split by word windows with configurable overlap."""

    def chunk(self, text: str) -> list[str]:
        words = text.split()
        if not words:
            return []

        step = self.config.chunk_size - self.config.chunk_overlap
        chunks = []
        for start in range(0, len(words), step):
            end = start + self.config.chunk_size
            chunks.append(" ".join(words[start:end]))
            if end >= len(words):
                break
        return chunks


class SentenceChunker(Chunker):
    """Group complete sentences until the target word size is reached."""

    sentence_pattern = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")

    def chunk(self, text: str) -> list[str]:
        sentences = [
            part.strip()
            for part in self.sentence_pattern.split(text.strip())
            if part.strip()
        ]
        if not sentences:
            return []

        chunks: list[str] = []
        current: list[str] = []
        current_words = 0
        for sentence in sentences:
            sentence_words = len(sentence.split())
            if current and current_words + sentence_words > self.config.chunk_size:
                chunks.append(" ".join(current))
                current = []
                current_words = 0
            current.append(sentence)
            current_words += sentence_words
        if current:
            chunks.append(" ".join(current))
        return chunks


class ParagraphChunker(Chunker):
    """Keep paragraphs intact, falling back to fixed windows for oversized paragraphs."""

    def chunk(self, text: str) -> list[str]:
        paragraphs = [
            part.strip() for part in re.split(r"\n\s*\n+", text.strip()) if part.strip()
        ]
        if not paragraphs:
            return []

        chunks: list[str] = []
        fallback = FixedSizeChunker(self.config)
        current: list[str] = []
        current_words = 0
        for paragraph in paragraphs:
            paragraph_words = len(paragraph.split())
            if paragraph_words > self.config.chunk_size:
                if current:
                    chunks.append("\n\n".join(current))
                    current = []
                    current_words = 0
                chunks.extend(fallback.chunk(paragraph))
                continue
            if current and current_words + paragraph_words > self.config.chunk_size:
                chunks.append("\n\n".join(current))
                current = []
                current_words = 0
            current.append(paragraph)
            current_words += paragraph_words
        if current:
            chunks.append("\n\n".join(current))
        return chunks


class RecursiveChunker(Chunker):
    """Split using paragraph, sentence, then word boundaries as needed."""

    separators = ("\n\n", "\n", ". ", " ")

    def chunk(self, text: str) -> list[str]:
        if not text.strip():
            return []
        chunks = self._split(text.strip(), 0)
        if self.config.chunk_overlap == 0:
            return chunks

        overlapped: list[str] = []
        for index, chunk in enumerate(chunks):
            if index > 0:
                previous_words = chunks[index - 1].split()
                overlap = previous_words[-self.config.chunk_overlap :]
                chunk = " ".join(overlap + chunk.split())
            overlapped.append(chunk)
        return overlapped

    def _split(self, text: str, separator_index: int) -> list[str]:
        if len(text.split()) <= self.config.chunk_size:
            return [text]
        if separator_index >= len(self.separators):
            return FixedSizeChunker(
                ChunkingConfig(
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=0,
                    semantic_threshold=self.config.semantic_threshold,
                )
            ).chunk(text)

        separator = self.separators[separator_index]
        parts = [part.strip() for part in text.split(separator) if part.strip()]
        if len(parts) <= 1:
            return self._split(text, separator_index + 1)

        chunks: list[str] = []
        current: list[str] = []
        current_words = 0
        for part in parts:
            part_words = len(part.split())
            if part_words > self.config.chunk_size:
                if current:
                    chunks.append(separator.join(current))
                    current = []
                    current_words = 0
                chunks.extend(self._split(part, separator_index + 1))
            elif current and current_words + part_words > self.config.chunk_size:
                chunks.append(separator.join(current))
                current = [part]
                current_words = part_words
            else:
                current.append(part)
                current_words += part_words
        if current:
            chunks.append(separator.join(current))
        return chunks


class SemanticChunker(Chunker):
    """Group adjacent sentences while their embedding similarity stays above a threshold."""

    sentence_pattern = SentenceChunker.sentence_pattern

    def __init__(
        self, config: ChunkingConfig, embedding_provider: EmbeddingProvider | None
    ):
        super().__init__(config)
        if embedding_provider is None:
            raise ValueError("semantic chunking requires an embedding provider")
        self.embedding_provider = embedding_provider

    def chunk(self, text: str) -> list[str]:
        sentences = [
            part.strip()
            for part in self.sentence_pattern.split(text.strip())
            if part.strip()
        ]
        if not sentences:
            return []
        if len(sentences) == 1:
            return sentences

        vectors = self.embedding_provider.embed_documents(sentences)
        chunks: list[str] = []
        current = [sentences[0]]
        current_words = len(sentences[0].split())
        for index in range(1, len(sentences)):
            similarity = _cosine_similarity(vectors[index - 1], vectors[index])
            sentence = sentences[index]
            sentence_words = len(sentence.split())
            boundary = similarity < self.config.semantic_threshold
            too_large = current_words + sentence_words > self.config.chunk_size
            if boundary or too_large:
                chunks.append(" ".join(current))
                current = [sentence]
                current_words = sentence_words
            else:
                current.append(sentence)
                current_words += sentence_words
        if current:
            chunks.append(" ".join(current))
        return chunks


def _cosine_similarity(first: list[float], second: list[float]) -> float:
    if len(first) != len(second):
        raise ValueError("Semantic chunking embeddings must have the same dimension.")
    first_magnitude = sum(value * value for value in first) ** 0.5
    second_magnitude = sum(value * value for value in second) ** 0.5
    if first_magnitude == 0 or second_magnitude == 0:
        return 0.0
    return sum(a * b for a, b in zip(first, second)) / (
        first_magnitude * second_magnitude
    )
