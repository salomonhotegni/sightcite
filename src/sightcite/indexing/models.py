"""Metadata models for persistent document indexes."""

from dataclasses import dataclass
from pathlib import Path
from string import hexdigits

INDEX_SCHEMA_VERSION = 1


def _validate_nonblank(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")


def _validate_positive(value: int, field_name: str) -> None:
    if value <= 0:
        raise ValueError(f"{field_name} must be greater than zero")


@dataclass(frozen=True, slots=True, kw_only=True)
class TextIndexConfiguration:
    """Configuration used to construct a persistent text index."""

    model_name: str
    dimension: int
    chunk_size: int
    overlap: int
    ocr_enabled: bool = False
    min_native_chars: int = 20
    ocr_dpi: int = 144

    def __post_init__(self) -> None:
        _validate_nonblank(self.model_name, "text model name")
        _validate_positive(self.dimension, "text embedding dimension")
        _validate_positive(self.chunk_size, "chunk size")

        if self.overlap < 0:
            raise ValueError("overlap must not be negative")

        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk size")

        if self.min_native_chars < 0:
            raise ValueError("minimum native characters must not be negative")

        _validate_positive(self.ocr_dpi, "OCR dpi")


@dataclass(frozen=True, slots=True, kw_only=True)
class VisualIndexConfiguration:
    """Configuration used to construct a persistent visual index."""

    model_name: str
    dimension: int
    dpi: int = 144

    def __post_init__(self) -> None:
        _validate_nonblank(self.model_name, "visual model name")
        _validate_positive(self.dimension, "visual embedding dimension")
        _validate_positive(self.dpi, "visual dpi")


@dataclass(frozen=True, slots=True, kw_only=True)
class DocumentIndexManifest:
    """Versioned metadata describing one persistent document index."""

    document_id: str
    source_filename: str
    source_sha256: str
    page_count: int
    text: TextIndexConfiguration
    visual: VisualIndexConfiguration
    schema_version: int = INDEX_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != INDEX_SCHEMA_VERSION:
            raise ValueError(f"unsupported index schema version: {self.schema_version}")

        _validate_nonblank(self.document_id, "document id")
        _validate_nonblank(self.source_filename, "source filename")

        if Path(self.source_filename).name != self.source_filename:
            raise ValueError("source filename must not contain a directory")

        if (
            len(self.source_sha256) != 64
            or self.source_sha256 != self.source_sha256.lower()
            or any(character not in hexdigits for character in self.source_sha256)
        ):
            raise ValueError("source sha256 must be a lowercase 64-character hexadecimal digest")

        _validate_positive(self.page_count, "page count")
