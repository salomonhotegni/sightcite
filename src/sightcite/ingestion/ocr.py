"""OCR fallback routing for text-deficient PDF pages."""

from pathlib import Path
from typing import Protocol

from sightcite.ingestion.models import ExtractedPage, ExtractionSource
from sightcite.ingestion.pdf import render_pdf
from sightcite.ingestion.text import extract_pdf_text


class OcrBackend(Protocol):
    """Interface implemented by OCR engines."""

    def extract_text(self, image_path: Path) -> str:
        """Extract text from one rendered page image."""
        ...


def extract_pdf_text_with_ocr(
    pdf_path: str | Path,
    output_dir: str | Path,
    backend: OcrBackend,
    *,
    min_native_chars: int = 20,
    dpi: int = 144,
) -> list[ExtractedPage]:
    """Extract native text and apply OCR only to deficient pages.

    Args:
        pdf_path: Source PDF.
        output_dir: Directory for page images required by OCR.
        backend: OCR engine implementation.
        min_native_chars: Minimum non-whitespace native characters required
            to avoid OCR.
        dpi: Page-image rendering resolution.

    Returns:
        Pages in document order with final text and extraction provenance.

    Raises:
        ValueError: If ``min_native_chars`` is negative.
    """
    if min_native_chars < 0:
        raise ValueError("min_native_chars must not be negative")

    native_pages = extract_pdf_text(pdf_path)
    deficient_pages = {
        page.page_number
        for page in native_pages
        if _non_whitespace_length(page.text) < min_native_chars
    }

    if not deficient_pages:
        return native_pages

    rendered_pages = render_pdf(
        pdf_path,
        output_dir,
        dpi=dpi,
    )
    images_by_page = {page.page_number: page.image_path for page in rendered_pages}

    extracted_pages: list[ExtractedPage] = []

    for page in native_pages:
        if page.page_number not in deficient_pages:
            extracted_pages.append(page)
            continue

        ocr_text = backend.extract_text(images_by_page[page.page_number]).strip()

        if ocr_text:
            extracted_pages.append(
                ExtractedPage(
                    page_number=page.page_number,
                    text=ocr_text,
                    source=ExtractionSource.OCR,
                )
            )
        else:
            extracted_pages.append(page)

    return extracted_pages


def _non_whitespace_length(text: str) -> int:
    return len("".join(text.split()))
