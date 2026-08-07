"""Rank fusion for page retrieval."""

from collections.abc import Mapping, Sequence
from math import fsum

from sightcite.retrieval.models import FusedPageResult

DEFAULT_RRF_CONSTANT = 60


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[int]],
    *,
    rank_constant: int = DEFAULT_RRF_CONSTANT,
    weights: Mapping[str, float] | None = None,
    top_k: int | None = None,
) -> list[FusedPageResult]:
    """Combine page rankings using weighted Reciprocal Rank Fusion.

    Duplicate pages within one source ranking are counted only once,
    at their first occurrence.
    """
    if not rankings:
        raise ValueError("at least one ranking source is required")

    if rank_constant < 0:
        raise ValueError("rank_constant must not be negative")

    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    for source_name in rankings:
        if not source_name.strip():
            raise ValueError("ranking source names must not be blank")

    if weights is None:
        resolved_weights = {source_name: 1.0 for source_name in rankings}
    else:
        if set(weights) != set(rankings):
            raise ValueError("weights must match ranking sources")

        resolved_weights = dict(weights)

    if any(weight <= 0 for weight in resolved_weights.values()):
        raise ValueError("ranking weights must be greater than zero")

    contributions: dict[int, list[float]] = {}
    source_ranks: dict[int, dict[str, int]] = {}

    for source_name, pages in rankings.items():
        seen_pages: set[int] = set()

        for page_number in pages:
            if page_number <= 0:
                raise ValueError("ranked page numbers must be positive")

            if page_number in seen_pages:
                continue

            seen_pages.add(page_number)
            unique_rank = len(seen_pages)
            contribution = resolved_weights[source_name] / (rank_constant + unique_rank)

            contributions.setdefault(
                page_number,
                [],
            ).append(contribution)
            source_ranks.setdefault(
                page_number,
                {},
            )[source_name] = unique_rank

    ordered_pages = sorted(
        contributions,
        key=lambda page_number: (
            -fsum(contributions[page_number]),
            min(source_ranks[page_number].values()),
            page_number,
        ),
    )

    if top_k is not None:
        ordered_pages = ordered_pages[:top_k]

    return [
        FusedPageResult(
            rank=rank,
            score=fsum(contributions[page_number]),
            page_number=page_number,
            source_ranks=tuple(sorted(source_ranks[page_number].items())),
        )
        for rank, page_number in enumerate(
            ordered_pages,
            start=1,
        )
    ]
