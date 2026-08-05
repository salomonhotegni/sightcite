"""Scientific-document ingestion."""

from sightcite.ingestion.chunking import chunk_extracted_pages
from sightcite.ingestion.models import (
    ExtractedPage,
    ExtractionSource,
    RenderedPage,
    TextChunk,
)
from sightcite.ingestion.ocr import OcrBackend, extract_pdf_text_with_ocr
from sightcite.ingestion.pdf import render_pdf
from sightcite.ingestion.tesseract import TesseractOcrBackend
from sightcite.ingestion.text import extract_pdf_text

__all__ = [
    "ExtractedPage",
    "ExtractionSource",
    "OcrBackend",
    "RenderedPage",
    "TesseractOcrBackend",
    "TextChunk",
    "chunk_extracted_pages",
    "extract_pdf_text",
    "extract_pdf_text_with_ocr",
    "render_pdf",
]
