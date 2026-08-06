import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from sightcite.evaluation import (
    RetrievalBenchmark,
    RetrievalExample,
    run_text_retrieval_benchmark,
    run_visual_retrieval_benchmark,
    write_benchmark_report,
)


class BenchmarkEmbedder:
    """Deterministic embedder for benchmark tests."""

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
        if "first" in query.lower():
            return np.asarray([1.0, 0.0], dtype=np.float64)

        return np.asarray([0.0, 1.0], dtype=np.float64)


class BenchmarkVisualEmbedder:
    """Deterministic visual embedder for benchmark tests."""

    @property
    def dimension(self) -> int:
        return 2

    def embed_images(
        self,
        image_paths: Sequence[Path],
    ) -> npt.NDArray[np.float64]:
        vectors = {
            "page_0001.png": [1.0, 0.0],
            "page_0002.png": [0.0, 1.0],
        }
        return np.asarray(
            [vectors[path.name] for path in image_paths],
            dtype=np.float64,
        )

    def embed_query(
        self,
        query: str,
    ) -> npt.NDArray[np.float64]:
        if "first" in query.lower():
            return np.asarray(
                [1.0, 0.0],
                dtype=np.float64,
            )

        return np.asarray(
            [0.0, 1.0],
            dtype=np.float64,
        )


class FakeOcrBackend:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[Path] = []

    def extract_text(self, image_path: Path) -> str:
        self.calls.append(image_path)
        return self.text


def make_benchmark(sample_pdf: Path) -> RetrievalBenchmark:
    return RetrievalBenchmark(
        schema_version=1,
        document_id="sample-paper",
        pdf_path=sample_pdf,
        examples=(
            RetrievalExample(
                query_id="q1",
                query="What is on the first page?",
                relevant_pages=frozenset({1}),
            ),
            RetrievalExample(
                query_id="q2",
                query="What is on the second page?",
                relevant_pages=frozenset({2}),
            ),
        ),
    )


def make_blank_benchmark(blank_pdf: Path) -> RetrievalBenchmark:
    return RetrievalBenchmark(
        schema_version=1,
        document_id="scanned-paper",
        pdf_path=blank_pdf,
        examples=(
            RetrievalExample(
                query_id="q1",
                query="What was recovered?",
                relevant_pages=frozenset({1}),
            ),
        ),
    )


def test_run_text_retrieval_benchmark(
    sample_pdf: Path,
) -> None:
    result = run_text_retrieval_benchmark(
        make_benchmark(sample_pdf),
        BenchmarkEmbedder(),
        system_name="test-text",
    )

    assert result.system_name == "test-text"
    assert result.document_id == "sample-paper"
    assert result.pdf_path == sample_pdf

    assert result.evaluation.metrics.query_count == 2
    assert result.evaluation.metrics.recall_at_1 == 1.0
    assert result.evaluation.metrics.recall_at_3 == 1.0
    assert result.evaluation.metrics.recall_at_5 == 1.0
    assert result.evaluation.metrics.mean_reciprocal_rank == 1.0


def test_write_benchmark_report(
    sample_pdf: Path,
    tmp_path: Path,
) -> None:
    result = run_text_retrieval_benchmark(
        make_benchmark(sample_pdf),
        BenchmarkEmbedder(),
    )
    output_path = tmp_path / "nested" / "report.json"

    returned_path = write_benchmark_report(result, output_path)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert returned_path == output_path
    assert payload["schema_version"] == 1
    assert payload["system_name"] == "bge-text"
    assert payload["document_id"] == "sample-paper"
    assert payload["metrics"]["recall_at_1"] == 1.0
    assert payload["metrics"]["mean_reciprocal_rank"] == 1.0
    assert payload["queries"][0]["query_id"] == "q1"
    assert payload["queries"][0]["relevant_pages"] == [1]
    assert payload["queries"][0]["retrieved_pages"] == [1, 2]


def test_benchmark_result_rejects_blank_system_name(
    sample_pdf: Path,
) -> None:
    with pytest.raises(ValueError, match="system_name must not be blank"):
        run_text_retrieval_benchmark(
            make_benchmark(sample_pdf),
            BenchmarkEmbedder(),
            system_name=" ",
        )


def test_run_text_retrieval_benchmark_with_ocr(
    blank_pdf: Path,
    tmp_path: Path,
) -> None:
    backend = FakeOcrBackend("Text recovered from scanned paper")
    output_dir = tmp_path / "ocr-pages"

    result = run_text_retrieval_benchmark(
        make_blank_benchmark(blank_pdf),
        BenchmarkEmbedder(),
        system_name="bge-text-ocr",
        ocr_backend=backend,
        ocr_output_dir=output_dir,
        min_native_chars=25,
        ocr_dpi=200,
    )

    assert result.system_name == "bge-text-ocr"
    assert result.evaluation.metrics.query_count == 1
    assert result.evaluation.metrics.recall_at_1 == 1.0
    assert len(backend.calls) == 1
    assert backend.calls[0].parent == output_dir
    assert backend.calls[0].is_file()


def test_run_visual_retrieval_benchmark(
    sample_pdf: Path,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "visual-pages"

    result = run_visual_retrieval_benchmark(
        make_benchmark(sample_pdf),
        BenchmarkVisualEmbedder(),
        system_name="test-visual",
        output_dir=output_dir,
        dpi=100,
    )

    assert result.system_name == "test-visual"
    assert result.document_id == "sample-paper"
    assert result.pdf_path == sample_pdf
    assert result.evaluation.metrics.query_count == 2
    assert result.evaluation.metrics.recall_at_1 == 1.0
    assert result.evaluation.metrics.recall_at_3 == 1.0
    assert result.evaluation.metrics.recall_at_5 == 1.0
    assert result.evaluation.metrics.mean_reciprocal_rank == 1.0
    assert (output_dir / "page_0001.png").is_file()
    assert (output_dir / "page_0002.png").is_file()
