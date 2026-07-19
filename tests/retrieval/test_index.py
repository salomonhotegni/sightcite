import numpy as np
import pytest

from sightcite.ingestion import TextChunk
from sightcite.retrieval import TextVectorIndex


def make_chunks() -> list[TextChunk]:
    return [
        TextChunk(1, 0, "cats", 0, 1),
        TextChunk(2, 0, "dogs", 0, 1),
        TextChunk(3, 0, "cats and dogs", 0, 3),
    ]


def test_search_ranks_chunks_by_cosine_similarity() -> None:
    chunks = make_chunks()
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.8, 0.6],
        ]
    )
    index = TextVectorIndex(chunks, embeddings)

    results = index.search([1.0, 0.0], top_k=3)

    assert index.size == 3
    assert index.dimension == 2
    assert [result.rank for result in results] == [1, 2, 3]
    assert [result.chunk.text for result in results] == [
        "cats",
        "cats and dogs",
        "dogs",
    ]
    assert [result.score for result in results] == pytest.approx([1.0, 0.8, 0.0])


def test_search_is_invariant_to_query_magnitude() -> None:
    index = TextVectorIndex(
        make_chunks(),
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.8, 0.6],
        ],
    )

    unit_results = index.search([1.0, 0.0])
    scaled_results = index.search([10.0, 0.0])

    assert [result.chunk for result in unit_results] == [result.chunk for result in scaled_results]
    assert [result.score for result in unit_results] == pytest.approx(
        [result.score for result in scaled_results]
    )


def test_search_limits_results_to_available_chunks() -> None:
    index = TextVectorIndex(
        make_chunks()[:2],
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
    )

    results = index.search([1.0, 0.0], top_k=10)

    assert len(results) == 2


def test_search_empty_index_returns_no_results() -> None:
    index = TextVectorIndex([], np.empty((0, 2)))

    assert index.size == 0
    assert index.search([1.0, 0.0]) == []


def test_index_rejects_non_matrix_embeddings() -> None:
    with pytest.raises(
        ValueError,
        match="embeddings must be a two-dimensional matrix",
    ):
        TextVectorIndex(make_chunks(), [1.0, 2.0, 3.0])


def test_index_rejects_zero_embedding_dimension() -> None:
    with pytest.raises(
        ValueError,
        match="embeddings must have at least one dimension",
    ):
        TextVectorIndex([], np.empty((0, 0)))


def test_index_rejects_chunk_count_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="number of embeddings must match",
    ):
        TextVectorIndex(make_chunks(), np.ones((2, 3)))


def test_index_rejects_non_finite_embeddings() -> None:
    embeddings = np.array(
        [
            [1.0, 0.0],
            [np.nan, 1.0],
            [0.8, 0.6],
        ]
    )

    with pytest.raises(ValueError, match="only finite values"):
        TextVectorIndex(make_chunks(), embeddings)


def test_index_rejects_zero_embedding() -> None:
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 0.0],
            [0.8, 0.6],
        ]
    )

    with pytest.raises(ValueError, match="embedding vectors must not be zero"):
        TextVectorIndex(make_chunks(), embeddings)


def test_search_rejects_non_positive_top_k() -> None:
    index = TextVectorIndex(make_chunks(), np.eye(3))

    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        index.search([1.0, 0.0, 0.0], top_k=0)


def test_search_rejects_non_vector_query() -> None:
    index = TextVectorIndex(make_chunks(), np.eye(3))

    with pytest.raises(
        ValueError,
        match="query_embedding must be one-dimensional",
    ):
        index.search([[1.0, 0.0, 0.0]])


def test_search_rejects_wrong_query_dimension() -> None:
    index = TextVectorIndex(make_chunks(), np.eye(3))

    with pytest.raises(ValueError, match="does not match the index dimension"):
        index.search([1.0, 0.0])


def test_search_rejects_non_finite_query() -> None:
    index = TextVectorIndex(make_chunks(), np.eye(3))

    with pytest.raises(ValueError, match="only finite values"):
        index.search([1.0, np.inf, 0.0])


def test_search_rejects_zero_query() -> None:
    index = TextVectorIndex(make_chunks(), np.eye(3))

    with pytest.raises(ValueError, match="query_embedding must not be zero"):
        index.search([0.0, 0.0, 0.0])
