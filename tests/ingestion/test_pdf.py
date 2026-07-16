from pathlib import Path

import pymupdf
import pytest

from sightcite.ingestion import render_pdf


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "sample.pdf"

    with pymupdf.open() as document:  # type: ignore[no-untyped-call]
        for text in ("First page", "Second page"):
            page = document.new_page(width=300, height=200)
            page.insert_text((72, 72), text)

        document.save(pdf_path)

    return pdf_path


def test_render_pdf_creates_one_image_per_page(
    sample_pdf: Path,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "pages"

    pages = render_pdf(sample_pdf, output_dir, dpi=144)

    assert len(pages) == 2
    assert [page.page_number for page in pages] == [1, 2]
    assert [page.image_path.name for page in pages] == [
        "page_0001.png",
        "page_0002.png",
    ]

    for page in pages:
        assert page.image_path.is_file()
        assert page.width == 600
        assert page.height == 400


def test_render_pdf_creates_nested_output_directory(
    sample_pdf: Path,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "nested" / "pages"

    pages = render_pdf(sample_pdf, output_dir)

    assert output_dir.is_dir()
    assert len(pages) == 2


def test_render_pdf_rejects_missing_source(tmp_path: Path) -> None:
    missing_pdf = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError, match="PDF file does not exist"):
        render_pdf(missing_pdf, tmp_path / "pages")


def test_render_pdf_rejects_non_positive_dpi(
    sample_pdf: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="dpi must be greater than zero"):
        render_pdf(sample_pdf, tmp_path / "pages", dpi=0)
