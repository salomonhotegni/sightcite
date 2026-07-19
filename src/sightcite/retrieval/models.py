"""Data models produced by retrieval."""

from dataclasses import dataclass

from sightcite.ingestion import TextChunk


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One ranked text-retrieval result."""

    rank: int
    score: float
    chunk: TextChunk
