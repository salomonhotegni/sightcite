import json
from pathlib import Path

import pytest

from sightcite.evaluation import load_retrieval_benchmark


def write_dataset(
    path: Path,
    *,
    pdf_path: str = "sample.pdf",
    questions: list[dict[str, object]] | None = None,
    schema_version: object = 1,
) -> None:
    if questions is None:
        questions = [
            {
                "query_id": "q1",
                "query": "What is on the first page?",
                "relevant_pages": [1],
            },
            {
                "query_id": "q2",
                "query": "What is on the second page?",
                "relevant_pages": [2],
            },
        ]

    payload = {
        "schema_version": schema_version,
        "document_id": "sample-paper",
        "pdf_path": pdf_path,
        "questions": questions,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_retrieval_benchmark_resolves_relative_pdf(
    sample_pdf: Path,
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "benchmark.json"
    write_dataset(dataset_path)

    benchmark = load_retrieval_benchmark(dataset_path)

    assert benchmark.schema_version == 1
    assert benchmark.document_id == "sample-paper"
    assert benchmark.pdf_path == sample_pdf.resolve()
    assert len(benchmark.examples) == 2
    assert benchmark.examples[0].query_id == "q1"
    assert benchmark.examples[0].relevant_pages == frozenset({1})


def test_load_retrieval_benchmark_accepts_absolute_pdf(
    sample_pdf: Path,
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "benchmark.json"
    write_dataset(dataset_path, pdf_path=str(sample_pdf))

    benchmark = load_retrieval_benchmark(dataset_path)

    assert benchmark.pdf_path == sample_pdf.resolve()


def test_load_retrieval_benchmark_rejects_missing_dataset(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="dataset does not exist"):
        load_retrieval_benchmark(tmp_path / "missing.json")


def test_load_retrieval_benchmark_rejects_invalid_json(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "benchmark.json"
    dataset_path.write_text("{invalid", encoding="utf-8")

    with pytest.raises(ValueError, match="contains invalid JSON"):
        load_retrieval_benchmark(dataset_path)


def test_load_retrieval_benchmark_rejects_missing_pdf(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "benchmark.json"
    write_dataset(dataset_path, pdf_path="missing.pdf")

    with pytest.raises(FileNotFoundError, match="benchmark PDF does not exist"):
        load_retrieval_benchmark(dataset_path)


def test_load_retrieval_benchmark_rejects_duplicate_query_ids(
    sample_pdf: Path,
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "benchmark.json"
    questions: list[dict[str, object]] = [
        {
            "query_id": "duplicate",
            "query": "First question",
            "relevant_pages": [1],
        },
        {
            "query_id": "duplicate",
            "query": "Second question",
            "relevant_pages": [2],
        },
    ]
    write_dataset(dataset_path, questions=questions)

    with pytest.raises(ValueError, match="query IDs must be unique"):
        load_retrieval_benchmark(dataset_path)


@pytest.mark.parametrize(
    ("schema_version", "message"),
    [
        (2, "unsupported benchmark schema version"),
        ("1", "schema_version must be an integer"),
        (True, "schema_version must be an integer"),
    ],
)
def test_load_retrieval_benchmark_rejects_invalid_schema_version(
    sample_pdf: Path,
    tmp_path: Path,
    schema_version: object,
    message: str,
) -> None:
    dataset_path = tmp_path / "benchmark.json"
    write_dataset(dataset_path, schema_version=schema_version)

    with pytest.raises(ValueError, match=message):
        load_retrieval_benchmark(dataset_path)


def test_load_retrieval_benchmark_rejects_non_integer_page(
    sample_pdf: Path,
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "benchmark.json"
    questions: list[dict[str, object]] = [
        {
            "query_id": "q1",
            "query": "Question",
            "relevant_pages": ["1"],
        }
    ]
    write_dataset(dataset_path, questions=questions)

    with pytest.raises(ValueError, match="relevant_pages must be an integer"):
        load_retrieval_benchmark(dataset_path)
