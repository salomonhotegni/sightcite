from pathlib import Path

import numpy as np
import pytest

from sightcite.ingestion import RenderedPage
from sightcite.retrieval import VisualVectorIndex


def make_pages() -> list[RenderedPage]:
    return [
        RenderedPage(1, Path("page-1.png"), 600, 800),
        RenderedPage(2, Path("page-2.png"), 600, 800),
        RenderedPage(3, Path("page-3.png"), 600, 800),
    ]


def test_visual_index_ranks_pages_by_cosine_similarity() -> None:
    index = VisualVectorIndex(
        make_pages(),
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.8, 0.6],
        ],
    )

    results = index.search(
        [1.0, 0.0],
        top_k=3,
    )

    assert index.size == 3
    assert index.dimension == 2
    assert [result.rank for result in results] == [1, 2, 3]
    assert [result.page.page_number for result in results] == [1, 3, 2]
    assert [result.score for result in results] == pytest.approx([1.0, 0.8, 0.0])


def test_visual_index_limits_results_to_available_pages() -> None:
    index = VisualVectorIndex(
        make_pages()[:2],
        np.eye(2),
    )

    results = index.search(
        [1.0, 0.0],
        top_k=10,
    )

    assert len(results) == 2


def test_visual_index_supports_empty_index() -> None:
    index = VisualVectorIndex(
        [],
        np.empty((0, 2)),
    )

    assert index.size == 0
    assert index.search([1.0, 0.0]) == []


def test_visual_index_rejects_page_count_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="number of embeddings must match the number of pages",
    ):
        VisualVectorIndex(
            make_pages(),
            np.ones((2, 3)),
        )
