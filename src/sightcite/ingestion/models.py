"""Data models produced by document ingestion."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RenderedPage:
    """Metadata describing one rendered PDF page."""

    page_number: int
    image_path: Path
    width: int
    height: int


class ExtractionSource(StrEnum):
    """Method that produced a page's final text."""

    NATIVE = "native"
    OCR = "ocr"


@dataclass(frozen=True, slots=True)
class ExtractedPage:
    """Text extracted from one PDF page."""

    page_number: int
    text: str
    source: ExtractionSource = ExtractionSource.NATIVE


@dataclass(frozen=True, slots=True)
class TextChunk:
    """Retrieval-ready text from one PDF page."""

    page_number: int
    chunk_index: int
    text: str
    start_word: int
    end_word: int
