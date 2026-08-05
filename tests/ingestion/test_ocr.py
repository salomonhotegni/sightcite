from pathlib import Path

import pymupdf
import pytest

from sightcite.ingestion import (
    ExtractionSource,
    extract_pdf_text_with_ocr,
)


class FakeOcrBackend:
    def __init__(self, results: dict[str, str]) -> None:
        self.results = results
        self.calls: list[Path] = []

    def extract_text(self, image_path: Path) -> str:
        self.calls.append(image_path)
        return self.results[image_path.name]


@pytest.fixture
def mixed_text_pdf(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "mixed.pdf"

    with pymupdf.open() as document:  # type: ignore[no-untyped-call]
        first_page = document.new_page(width=300, height=200)
        first_page.insert_text(
            (72, 72),
            "This page contains sufficient native text.",
        )

        document.new_page(width=300, height=200)

        third_page = document.new_page(width=300, height=200)
        third_page.insert_text((72, 72), "Short")

        document.save(pdf_path)

    return pdf_path


def test_ocr_runs_only_for_text_deficient_pages(
    mixed_text_pdf: Path,
    tmp_path: Path,
) -> None:
    backend = FakeOcrBackend(
        {
            "page_0002.png": "Text recovered with OCR",
            "page_0003.png": "   ",
        }
    )

    pages = extract_pdf_text_with_ocr(
        mixed_text_pdf,
        tmp_path / "rendered",
        backend,
        min_native_chars=10,
    )

    assert [page.page_number for page in pages] == [1, 2, 3]
    assert [page.text for page in pages] == [
        "This page contains sufficient native text.",
        "Text recovered with OCR",
        "Short",
    ]
    assert [page.source for page in pages] == [
        ExtractionSource.NATIVE,
        ExtractionSource.OCR,
        ExtractionSource.NATIVE,
    ]

    assert [path.name for path in backend.calls] == [
        "page_0002.png",
        "page_0003.png",
    ]
    assert all(path.is_file() for path in backend.calls)


def test_ocr_does_not_render_when_native_text_is_sufficient(
    sample_pdf: Path,
    tmp_path: Path,
) -> None:
    backend = FakeOcrBackend({})
    output_dir = tmp_path / "rendered"

    pages = extract_pdf_text_with_ocr(
        sample_pdf,
        output_dir,
        backend,
        min_native_chars=1,
    )

    assert len(pages) == 2
    assert backend.calls == []
    assert not output_dir.exists()
    assert all(page.source is ExtractionSource.NATIVE for page in pages)


def test_zero_threshold_disables_ocr(
    blank_pdf: Path,
    tmp_path: Path,
) -> None:
    backend = FakeOcrBackend({})

    pages = extract_pdf_text_with_ocr(
        blank_pdf,
        tmp_path / "rendered",
        backend,
        min_native_chars=0,
    )

    assert pages[0].text == ""
    assert pages[0].source is ExtractionSource.NATIVE
    assert backend.calls == []


def test_ocr_rejects_negative_native_character_threshold(
    sample_pdf: Path,
    tmp_path: Path,
) -> None:
    backend = FakeOcrBackend({})

    with pytest.raises(
        ValueError,
        match="min_native_chars must not be negative",
    ):
        extract_pdf_text_with_ocr(
            sample_pdf,
            tmp_path / "rendered",
            backend,
            min_native_chars=-1,
        )
