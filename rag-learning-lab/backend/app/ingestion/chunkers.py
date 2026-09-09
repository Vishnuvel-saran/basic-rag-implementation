from __future__ import annotations

import re
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.embeddings.base import EmbeddingProvider
    from app.llm.base import LLMProvider


logger = logging.getLogger(__name__)


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


@dataclass(frozen=True)
class ChunkSpan:
    """A chunk's text plus the exact word-index range it covers in the input.

    start/end are indices into `text.split()` of the string originally passed
    to `chunk_with_offsets`. end is exclusive. These are produced directly by
    the splitting logic, not reconstructed afterwards by searching for the
    text again -- that search is what silently misattributes pages when the
    same phrase (headers, footers, boilerplate) appears more than once.
    """

    text: str
    start: int
    end: int


class Chunker(ABC):
    """Strategy boundary: text in, ordered chunk spans out."""

    def __init__(self, config: ChunkingConfig):
        self.config = config

    @abstractmethod
    def chunk_with_offsets(self, text: str) -> list[ChunkSpan]:
        raise NotImplementedError

    def chunk(self, text: str) -> list[str]:
        """Back-compat convenience: text-only view of chunk_with_offsets."""
        return [span.text for span in self.chunk_with_offsets(text)]


class FixedSizeChunker(Chunker):
    """Split by word windows with configurable overlap."""

    def chunk_with_offsets(self, text: str) -> list[ChunkSpan]:
        words = text.split()
        if not words:
            return []

        step = self.config.chunk_size - self.config.chunk_overlap
        spans: list[ChunkSpan] = []
        for start in range(0, len(words), step):
            end = min(start + self.config.chunk_size, len(words))
            spans.append(ChunkSpan(" ".join(words[start:end]), start, end))
            if start + self.config.chunk_size >= len(words):
                break
        return spans


class SentenceChunker(Chunker):
    """Group complete sentences until the target word size is reached."""

    sentence_pattern = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")

    def chunk_with_offsets(self, text: str) -> list[ChunkSpan]:
        sentences = [
            part.strip()
            for part in self.sentence_pattern.split(text.strip())
            if part.strip()
        ]
        if not sentences:
            return []

        spans: list[ChunkSpan] = []
        current: list[str] = []
        current_words = 0
        chunk_start = 0
        cursor = 0
        for sentence in sentences:
            sentence_words = len(sentence.split())
            if current and current_words + sentence_words > self.config.chunk_size:
                spans.append(ChunkSpan(" ".join(current), chunk_start, cursor))
                current = []
                current_words = 0
                chunk_start = cursor
            current.append(sentence)
            current_words += sentence_words
            cursor += sentence_words
        if current:
            spans.append(ChunkSpan(" ".join(current), chunk_start, cursor))
        return spans


class ParagraphChunker(Chunker):
    """Keep paragraphs intact, falling back to fixed windows for oversized paragraphs."""

    def chunk_with_offsets(self, text: str) -> list[ChunkSpan]:
        paragraphs = [
            part.strip() for part in re.split(r"\n\s*\n+", text.strip()) if part.strip()
        ]
        if not paragraphs:
            return []

        spans: list[ChunkSpan] = []
        fallback = FixedSizeChunker(self.config)
        current: list[str] = []
        current_words = 0
        chunk_start = 0
        cursor = 0
        for paragraph in paragraphs:
            paragraph_words = len(paragraph.split())
            if paragraph_words > self.config.chunk_size:
                if current:
                    spans.append(ChunkSpan("\n\n".join(current), chunk_start, cursor))
                    current = []
                    current_words = 0
                for sub_span in fallback.chunk_with_offsets(paragraph):
                    spans.append(
                        ChunkSpan(
                            sub_span.text,
                            cursor + sub_span.start,
                            cursor + sub_span.end,
                        )
                    )
                cursor += paragraph_words
                chunk_start = cursor
                continue
            if current and current_words + paragraph_words > self.config.chunk_size:
                spans.append(ChunkSpan("\n\n".join(current), chunk_start, cursor))
                current = []
                current_words = 0
                chunk_start = cursor
            current.append(paragraph)
            current_words += paragraph_words
            cursor += paragraph_words
        if current:
            spans.append(ChunkSpan("\n\n".join(current), chunk_start, cursor))
        return spans


class RecursiveChunker(Chunker):
    """Split using paragraph, sentence, then word boundaries as needed."""

    separators = ("\n\n", "\n", ". ", " ")

    def chunk_with_offsets(self, text: str) -> list[ChunkSpan]:
        stripped = text.strip()
        if not stripped:
            return []
        spans = self._split(stripped, 0, 0)
        if self.config.chunk_overlap == 0:
            return spans

        overlapped: list[ChunkSpan] = []
        for index, span in enumerate(spans):
            if index == 0:
                overlapped.append(span)
                continue
            previous = spans[index - 1]
            previous_words = previous.text.split()
            overlap_count = min(self.config.chunk_overlap, len(previous_words))
            overlap_words = previous_words[len(previous_words) - overlap_count :]
            new_text = " ".join(overlap_words + span.text.split())
            # The overlap words are real words taken from just before this
            # span, so the true start is exactly overlap_count words earlier
            # -- no guessing, no re-searching the document for a match.
            overlapped.append(ChunkSpan(new_text, span.start - overlap_count, span.end))
        return overlapped

    def _split(self, text: str, separator_index: int, offset: int) -> list[ChunkSpan]:
        words = text.split()
        if len(words) <= self.config.chunk_size:
            return [ChunkSpan(text, offset, offset + len(words))]
        if separator_index >= len(self.separators):
            fallback = FixedSizeChunker(
                ChunkingConfig(
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=0,
                    semantic_threshold=self.config.semantic_threshold,
                )
            )
            return [
                ChunkSpan(span.text, offset + span.start, offset + span.end)
                for span in fallback.chunk_with_offsets(text)
            ]

        separator = self.separators[separator_index]
        parts = [part.strip() for part in text.split(separator) if part.strip()]
        if len(parts) <= 1:
            return self._split(text, separator_index + 1, offset)

        spans: list[ChunkSpan] = []
        current: list[str] = []
        current_words = 0
        chunk_start = offset
        cursor = offset
        for part in parts:
            part_words = len(part.split())
            if part_words > self.config.chunk_size:
                if current:
                    spans.append(
                        ChunkSpan(separator.join(current), chunk_start, cursor)
                    )
                    current = []
                    current_words = 0
                spans.extend(self._split(part, separator_index + 1, cursor))
                cursor += part_words
                chunk_start = cursor
            elif current and current_words + part_words > self.config.chunk_size:
                spans.append(ChunkSpan(separator.join(current), chunk_start, cursor))
                current = [part]
                current_words = part_words
                chunk_start = cursor
                cursor += part_words
            else:
                current.append(part)
                current_words += part_words
                cursor += part_words
        if current:
            spans.append(ChunkSpan(separator.join(current), chunk_start, cursor))
        return spans


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

    def chunk_with_offsets(self, text: str) -> list[ChunkSpan]:
        sentences = [
            part.strip()
            for part in self.sentence_pattern.split(text.strip())
            if part.strip()
        ]
        if not sentences:
            return []
        if len(sentences) == 1:
            word_count = len(sentences[0].split())
            return [ChunkSpan(sentences[0], 0, word_count)]

        vectors = self.embedding_provider.embed_documents(sentences)
        spans: list[ChunkSpan] = []
        current = [sentences[0]]
        current_words = len(sentences[0].split())
        chunk_start = 0
        cursor = current_words
        for index in range(1, len(sentences)):
            similarity = _cosine_similarity(vectors[index - 1], vectors[index])
            sentence = sentences[index]
            sentence_words = len(sentence.split())
            boundary = similarity < self.config.semantic_threshold
            too_large = current_words + sentence_words > self.config.chunk_size
            if boundary or too_large:
                spans.append(ChunkSpan(" ".join(current), chunk_start, cursor))
                current = [sentence]
                current_words = sentence_words
                chunk_start = cursor
            else:
                current.append(sentence)
                current_words += sentence_words
            cursor += sentence_words
        if current:
            spans.append(ChunkSpan(" ".join(current), chunk_start, cursor))
        return spans


class AgenticChunker(Chunker):
    """Use an LLM to choose boundaries among deterministic sentence boundaries.

    The model never writes chunk text. It only returns sentence positions that
    should start a new chunk, so every returned span remains an exact word range
    from the original input.
    """

    sentence_pattern = SentenceChunker.sentence_pattern
    max_window_sentences = 24

    def __init__(self, config: ChunkingConfig, llm_provider: LLMProvider | None):
        super().__init__(config)
        if llm_provider is None:
            raise ValueError("agentic chunking requires an LLM provider")
        self.llm_provider = llm_provider

    def chunk_with_offsets(self, text: str) -> list[ChunkSpan]:
        words = text.split()
        if not words:
            return []

        sentence_ranges = self._sentence_ranges(text)
        if not sentence_ranges:
            return []

        spans: list[ChunkSpan] = []
        for window_start in range(0, len(sentence_ranges), self.max_window_sentences):
            window_end = min(
                window_start + self.max_window_sentences, len(sentence_ranges)
            )
            window_ranges = sentence_ranges[window_start:window_end]
            boundaries = self._choose_boundaries(
                [self._words_slice(words, start, end) for start, end in window_ranges]
            )
            if boundaries is None:
                spans.extend(self._recursive_fallback(text, window_ranges))
                continue

            positions = [0, *boundaries, len(window_ranges)]
            for local_start, local_end in zip(positions, positions[1:]):
                start = window_ranges[local_start][0]
                end = window_ranges[local_end - 1][1]
                spans.append(
                    ChunkSpan(self._words_slice(words, start, end), start, end)
                )
        return spans

    def _choose_boundaries(self, sentences: list[str]) -> list[int] | None:
        allowed = list(range(1, len(sentences)))
        if not allowed:
            return []

        prompt = (
            "You are choosing boundaries for document chunking.\n"
            "Do not rewrite, summarize, or return any source text.\n"
            'Return only valid JSON in this shape: {"boundaries":[2,5]}.\n'
            "Each number means the sentence index where a new chunk starts. "
            f"Only choose from these allowed indices: {allowed}.\n"
            f"The soft target is about {self.config.chunk_size} words per chunk.\n\n"
            + "\n".join(
                f"Sentence {index}: {sentence}"
                for index, sentence in enumerate(sentences)
            )
        )
        try:
            response = self.llm_provider.generate(prompt)
            payload = self._parse_response(response)
            boundaries = payload.get("boundaries")
            if not isinstance(boundaries, list):
                raise ValueError("boundaries must be a list")
            if any(
                not isinstance(boundary, int) or boundary not in allowed
                for boundary in boundaries
            ):
                raise ValueError("response contains an invalid sentence boundary")
            return sorted(set(boundaries))
        except Exception as exc:
            logger.warning("Agentic chunking fell back to recursive splitting: %s", exc)
            return None

    @staticmethod
    def _parse_response(response: str) -> dict:
        response = response.strip()
        try:
            payload = json.loads(response)
        except json.JSONDecodeError:
            start = response.find("{")
            end = response.rfind("}")
            if start < 0 or end <= start:
                raise ValueError("LLM response was not JSON")
            payload = json.loads(response[start : end + 1])
        if not isinstance(payload, dict):
            raise ValueError("LLM response must be a JSON object")
        return payload

    def _recursive_fallback(
        self, text: str, sentence_ranges: list[tuple[int, int]]
    ) -> list[ChunkSpan]:
        words = text.split()
        start = sentence_ranges[0][0]
        end = sentence_ranges[-1][1]
        window_text = self._words_slice(words, start, end)
        fallback = RecursiveChunker(self.config).chunk_with_offsets(window_text)
        return [
            ChunkSpan(span.text, start + span.start, start + span.end)
            for span in fallback
        ]

    def _sentence_ranges(self, text: str) -> list[tuple[int, int]]:
        sentences = [
            part.strip()
            for part in self.sentence_pattern.split(text.strip())
            if part.strip()
        ]
        ranges: list[tuple[int, int]] = []
        cursor = 0
        for sentence in sentences:
            end = cursor + len(sentence.split())
            ranges.append((cursor, end))
            cursor = end
        return ranges

    @staticmethod
    def _words_slice(words: list[str], start: int, end: int) -> str:
        return " ".join(words[start:end])


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
