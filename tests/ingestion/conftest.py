from pathlib import Path

import pymupdf
import pytest


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Create a two-page PDF containing embedded text."""
    pdf_path = tmp_path / "sample.pdf"

    with pymupdf.open() as document:  # type: ignore[no-untyped-call]
        for text in ("First page", "Second page"):
            page = document.new_page(width=300, height=200)
            page.insert_text((72, 72), text)

        document.save(pdf_path)

    return pdf_path


@pytest.fixture
def blank_pdf(tmp_path: Path) -> Path:
    """Create a one-page PDF without embedded text."""
    pdf_path = tmp_path / "blank.pdf"

    with pymupdf.open() as document:  # type: ignore[no-untyped-call]
        document.new_page(width=300, height=200)
        document.save(pdf_path)

    return pdf_path
