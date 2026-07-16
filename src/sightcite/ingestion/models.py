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
