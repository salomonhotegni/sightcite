"""Evaluation data models."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RetrievalExample:
    """One page-retrieval evaluation question."""

    query_id: str
    query: str
    relevant_pages: frozenset[int]

    def __post_init__(self) -> None:
        if not self.query_id.strip():
            raise ValueError("query_id must not be blank")

        if not self.query.strip():
            raise ValueError("query must not be blank")

        if not self.relevant_pages:
            raise ValueError("relevant_pages must not be empty")

        if any(page <= 0 for page in self.relevant_pages):
            raise ValueError("relevant page numbers must be positive")


@dataclass(frozen=True, slots=True)
class QueryRetrievalEvaluation:
    """Metrics and ranked pages for one query."""

    query_id: str
    query: str
    relevant_pages: frozenset[int]
    retrieved_pages: tuple[int, ...]
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    reciprocal_rank: float


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    """Aggregate page-retrieval metrics."""

    query_count: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mean_reciprocal_rank: float


@dataclass(frozen=True, slots=True)
class RetrievalEvaluation:
    """Aggregate metrics and per-query evaluation records."""

    metrics: RetrievalMetrics
    queries: tuple[QueryRetrievalEvaluation, ...]


@dataclass(frozen=True, slots=True)
class RetrievalBenchmark:
    """Questions and page annotations for one PDF."""

    schema_version: int
    document_id: str
    pdf_path: Path
    examples: tuple[RetrievalExample, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported benchmark schema version")

        if not self.document_id.strip():
            raise ValueError("document_id must not be blank")

        if not self.pdf_path.is_file():
            raise FileNotFoundError(f"benchmark PDF does not exist: {self.pdf_path}")

        if not self.examples:
            raise ValueError("benchmark must contain at least one question")

        query_ids = [example.query_id for example in self.examples]

        if len(query_ids) != len(set(query_ids)):
            raise ValueError("benchmark query IDs must be unique")


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Results from evaluating one retrieval system on one document."""

    system_name: str
    document_id: str
    pdf_path: Path
    evaluation: RetrievalEvaluation

    def __post_init__(self) -> None:
        if not self.system_name.strip():
            raise ValueError("system_name must not be blank")
