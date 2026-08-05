"""Page-level retrieval evaluation."""

from collections.abc import Callable, Sequence
from math import fsum

from sightcite.evaluation.models import (
    QueryRetrievalEvaluation,
    RetrievalEvaluation,
    RetrievalExample,
    RetrievalMetrics,
)

PageRankingFunction = Callable[[str], Sequence[int]]


def evaluate_retrieval(
    examples: Sequence[RetrievalExample],
    rank_pages: PageRankingFunction,
) -> RetrievalEvaluation:
    """Evaluate ranked page retrieval over a collection of questions.

    Args:
        examples: Questions with relevant page-number annotations.
        rank_pages: Function returning page numbers in descending relevance.

    Returns:
        Aggregate metrics and per-query evaluation records.

    Raises:
        ValueError: If no examples are provided or a retriever returns an
            invalid page number.
    """
    if not examples:
        raise ValueError("at least one retrieval example is required")

    query_evaluations: list[QueryRetrievalEvaluation] = []

    for example in examples:
        retrieved_pages = _unique_pages(rank_pages(example.query))

        query_evaluations.append(
            QueryRetrievalEvaluation(
                query_id=example.query_id,
                query=example.query,
                relevant_pages=example.relevant_pages,
                retrieved_pages=retrieved_pages,
                recall_at_1=_recall_at_k(
                    retrieved_pages,
                    example.relevant_pages,
                    k=1,
                ),
                recall_at_3=_recall_at_k(
                    retrieved_pages,
                    example.relevant_pages,
                    k=3,
                ),
                recall_at_5=_recall_at_k(
                    retrieved_pages,
                    example.relevant_pages,
                    k=5,
                ),
                reciprocal_rank=_reciprocal_rank(
                    retrieved_pages,
                    example.relevant_pages,
                ),
            )
        )

    query_count = len(query_evaluations)

    metrics = RetrievalMetrics(
        query_count=query_count,
        recall_at_1=fsum(result.recall_at_1 for result in query_evaluations) / query_count,
        recall_at_3=fsum(result.recall_at_3 for result in query_evaluations) / query_count,
        recall_at_5=fsum(result.recall_at_5 for result in query_evaluations) / query_count,
        mean_reciprocal_rank=fsum(result.reciprocal_rank for result in query_evaluations)
        / query_count,
    )

    return RetrievalEvaluation(
        metrics=metrics,
        queries=tuple(query_evaluations),
    )


def _unique_pages(pages: Sequence[int]) -> tuple[int, ...]:
    unique_pages: list[int] = []
    seen: set[int] = set()

    for page in pages:
        if page <= 0:
            raise ValueError("retrieved page numbers must be positive")

        if page not in seen:
            seen.add(page)
            unique_pages.append(page)

    return tuple(unique_pages)


def _recall_at_k(
    retrieved_pages: Sequence[int],
    relevant_pages: frozenset[int],
    *,
    k: int,
) -> float:
    retrieved_at_k = set(retrieved_pages[:k])
    relevant_retrieved = len(retrieved_at_k & relevant_pages)

    return relevant_retrieved / len(relevant_pages)


def _reciprocal_rank(
    retrieved_pages: Sequence[int],
    relevant_pages: frozenset[int],
) -> float:
    for rank, page in enumerate(retrieved_pages, start=1):
        if page in relevant_pages:
            return 1.0 / rank

    return 0.0
