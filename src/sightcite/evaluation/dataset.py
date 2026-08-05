"""Retrieval benchmark dataset loading."""

import json
from pathlib import Path
from typing import cast

from sightcite.evaluation.models import RetrievalBenchmark, RetrievalExample


def load_retrieval_benchmark(
    dataset_path: str | Path,
) -> RetrievalBenchmark:
    """Load and validate a page-retrieval benchmark from JSON."""
    source = Path(dataset_path)

    if not source.is_file():
        raise FileNotFoundError(f"benchmark dataset does not exist: {source}")

    try:
        raw_payload: object = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"benchmark dataset contains invalid JSON: {source}") from error

    payload = _require_mapping(raw_payload, "benchmark root")
    schema_version = _require_integer(
        payload.get("schema_version"),
        "schema_version",
    )
    document_id = _require_string(
        payload.get("document_id"),
        "document_id",
    )
    pdf_value = _require_string(
        payload.get("pdf_path"),
        "pdf_path",
    )
    question_values = _require_list(
        payload.get("questions"),
        "questions",
    )

    pdf_path = Path(pdf_value)

    if not pdf_path.is_absolute():
        pdf_path = source.parent / pdf_path

    examples = tuple(
        _parse_question(value, index=index) for index, value in enumerate(question_values)
    )

    return RetrievalBenchmark(
        schema_version=schema_version,
        document_id=document_id,
        pdf_path=pdf_path.resolve(),
        examples=examples,
    )


def _parse_question(
    value: object,
    *,
    index: int,
) -> RetrievalExample:
    context = f"questions[{index}]"
    question = _require_mapping(value, context)

    query_id = _require_string(
        question.get("query_id"),
        f"{context}.query_id",
    )
    query = _require_string(
        question.get("query"),
        f"{context}.query",
    )
    page_values = _require_list(
        question.get("relevant_pages"),
        f"{context}.relevant_pages",
    )

    relevant_pages = frozenset(
        _require_integer(page, f"{context}.relevant_pages") for page in page_values
    )

    return RetrievalExample(
        query_id=query_id,
        query=query,
        relevant_pages=relevant_pages,
    )


def _require_mapping(
    value: object,
    field: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")

    return cast(dict[str, object], value)


def _require_list(
    value: object,
    field: str,
) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON array")

    return cast(list[object], value)


def _require_string(
    value: object,
    field: str,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")

    return value


def _require_integer(
    value: object,
    field: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")

    return value
