from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from sightcite.ingestion import ExtractionSource
from sightcite.pipelines import TextRetrievalPipeline


class KeywordEmbedder:
    """Deterministic embedder for pipeline tests."""

    @property
    def dimension(self) -> int:
        return 2

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> npt.NDArray[np.float64]:
        vectors = [[1.0, 0.0] if "First" in text else [0.0, 1.0] for text in texts]
        return np.asarray(vectors, dtype=np.float64)

    def embed_query(self, query: str) -> npt.NDArray[np.float64]:
        if query == "second":
            return np.asarray([0.0, 1.0], dtype=np.float64)

        return np.asarray([1.0, 0.0], dtype=np.float64)


class FakeOcrBackend:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[Path] = []

    def extract_text(self, image_path: Path) -> str:
        self.calls.append(image_path)
        return self.text


def test_text_pipeline_indexes_and_searches_pdf(sample_pdf: Path) -> None:
    pipeline = TextRetrievalPipeline(
        sample_pdf,
        KeywordEmbedder(),
        chunk_size=20,
        overlap=5,
    )

    results = pipeline.search("second", top_k=1)

    assert pipeline.source == sample_pdf
    assert pipeline.page_count == 2
    assert pipeline.chunk_count == 2
    assert len(pipeline.pages) == 2
    assert len(pipeline.chunks) == 2

    assert len(results) == 1
    assert results[0].rank == 1
    assert results[0].chunk.page_number == 2
    assert results[0].chunk.text == "Second page"


def test_text_pipeline_rejects_pdf_without_native_text(
    blank_pdf: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="PDF contains no extractable text",
    ):
        TextRetrievalPipeline(blank_pdf, KeywordEmbedder())


def test_text_pipeline_accepts_string_path(sample_pdf: Path) -> None:
    pipeline = TextRetrievalPipeline(str(sample_pdf), KeywordEmbedder())

    assert pipeline.source == sample_pdf


def test_text_pipeline_indexes_ocr_text(
    blank_pdf: Path,
    tmp_path: Path,
) -> None:
    backend = FakeOcrBackend("Text recovered from scanned page")

    pipeline = TextRetrievalPipeline(
        blank_pdf,
        KeywordEmbedder(),
        ocr_backend=backend,
        ocr_output_dir=tmp_path / "rendered",
    )

    assert pipeline.page_count == 1
    assert pipeline.chunk_count == 1
    assert pipeline.pages[0].text == "Text recovered from scanned page"
    assert pipeline.pages[0].source is ExtractionSource.OCR
    assert len(backend.calls) == 1
    assert backend.calls[0].is_file()


def test_text_pipeline_uses_temporary_ocr_images(
    blank_pdf: Path,
) -> None:
    backend = FakeOcrBackend("Temporary OCR text")

    pipeline = TextRetrievalPipeline(
        blank_pdf,
        KeywordEmbedder(),
        ocr_backend=backend,
    )

    assert pipeline.pages[0].source is ExtractionSource.OCR
    assert len(backend.calls) == 1
    assert not backend.calls[0].exists()


def test_text_pipeline_rejects_empty_ocr_result(
    blank_pdf: Path,
) -> None:
    backend = FakeOcrBackend("   ")

    with pytest.raises(
        ValueError,
        match="PDF contains no extractable text",
    ):
        TextRetrievalPipeline(
            blank_pdf,
            KeywordEmbedder(),
            ocr_backend=backend,
        )
