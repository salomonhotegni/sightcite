"""Evaluation data models."""

from dataclasses import dataclass


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
