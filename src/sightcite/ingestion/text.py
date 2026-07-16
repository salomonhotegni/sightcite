"""Native PDF text extraction."""

from pathlib import Path

import pymupdf

from sightcite.ingestion.models import ExtractedPage


def extract_pdf_text(pdf_path: str | Path) -> list[ExtractedPage]:
    """Extract embedded text from every page of a PDF.

    Args:
        pdf_path: Path to the source PDF.

    Returns:
        Extracted pages ordered by their one-based page numbers. Pages without
        embedded text are retained with an empty string.

    Raises:
        FileNotFoundError: If the source PDF does not exist.
    """
    source = Path(pdf_path)

    if not source.is_file():
        raise FileNotFoundError(f"PDF file does not exist: {source}")

    extracted_pages: list[ExtractedPage] = []

    with pymupdf.open(source) as document:  # type: ignore[no-untyped-call]
        for page_index, page in enumerate(document):
            # sort=True asks PyMuPDF to return text in a more natural reading order.
            text: str = page.get_text("text", sort=True)

            extracted_pages.append(
                ExtractedPage(
                    page_number=page_index + 1,
                    text=text.strip(),
                )
            )

    return extracted_pages
