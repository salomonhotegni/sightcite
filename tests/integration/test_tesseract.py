import os
from pathlib import Path

import pymupdf
import pytest

from sightcite.ingestion import TesseractOcrBackend, render_pdf

pytestmark = pytest.mark.ocr


@pytest.mark.skipif(
    os.environ.get("SIGHTCITE_RUN_OCR_TESTS") != "1",
    reason="Set SIGHTCITE_RUN_OCR_TESTS=1 to run Tesseract tests",
)
def test_tesseract_extracts_text_from_rendered_page(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "scanned.pdf"

    with pymupdf.open() as document:  # type: ignore[no-untyped-call]
        page = document.new_page(width=600, height=300)
        page.insert_text(
            (60, 150),
            "SIGHTCITE OCR TEST",
            fontsize=32,
        )
        document.save(pdf_path)

    rendered_pages = render_pdf(
        pdf_path,
        tmp_path / "rendered",
        dpi=300,
    )

    backend = TesseractOcrBackend()
    text = backend.extract_text(rendered_pages[0].image_path)

    assert "SIGHTCITE" in text.upper()
    assert "OCR" in text.upper()
    assert "TEST" in text.upper()
