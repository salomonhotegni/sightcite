"""Data models produced by document ingestion."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RenderedPage:
    """Metadata describing one rendered PDF page."""

    page_number: int
    image_path: Path
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class ExtractedPage:
    """Native text extracted from one PDF page."""

    page_number: int
    text: str


@dataclass(frozen=True, slots=True)
class TextChunk:
    """Retrieval-ready text from one PDF page."""

    page_number: int
    chunk_index: int
    text: str
    start_word: int
    end_word: int
