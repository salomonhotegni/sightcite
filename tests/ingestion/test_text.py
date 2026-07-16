from pathlib import Path

import pytest

from sightcite.ingestion import extract_pdf_text


def test_extract_pdf_text_preserves_page_order(sample_pdf: Path) -> None:
    pages = extract_pdf_text(sample_pdf)

    assert [page.page_number for page in pages] == [1, 2]
    assert [page.text for page in pages] == [
        "First page",
        "Second page",
    ]


def test_extract_pdf_text_retains_blank_pages(blank_pdf: Path) -> None:
    pages = extract_pdf_text(blank_pdf)

    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert pages[0].text == ""


def test_extract_pdf_text_accepts_string_path(sample_pdf: Path) -> None:
    pages = extract_pdf_text(str(sample_pdf))

    assert len(pages) == 2


def test_extract_pdf_text_rejects_missing_source(tmp_path: Path) -> None:
    missing_pdf = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError, match="PDF file does not exist"):
        extract_pdf_text(missing_pdf)
