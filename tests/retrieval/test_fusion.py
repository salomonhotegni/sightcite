import pytest

from sightcite.retrieval import reciprocal_rank_fusion


def test_rrf_combines_page_rankings() -> None:
    results = reciprocal_rank_fusion(
        {
            "text": [1, 2, 3],
            "visual": [2, 3, 1],
        }
    )

    assert [result.page_number for result in results] == [
        2,
        1,
        3,
    ]
    assert [result.rank for result in results] == [1, 2, 3]
    assert results[0].source_ranks == (
        ("text", 2),
        ("visual", 1),
    )


def test_rrf_deduplicates_pages_within_each_source() -> None:
    results = reciprocal_rank_fusion(
        {
            "text": [1, 1, 2],
            "visual": [2, 1],
        },
        rank_constant=0,
    )

    assert [result.page_number for result in results] == [1, 2]
    assert results[0].source_ranks == (
        ("text", 1),
        ("visual", 2),
    )
    assert results[0].score == pytest.approx(1.5)


def test_rrf_supports_source_weights() -> None:
    results = reciprocal_rank_fusion(
        {
            "text": [1, 2],
            "visual": [2, 1],
        },
        rank_constant=0,
        weights={
            "text": 2.0,
            "visual": 1.0,
        },
    )

    assert [result.page_number for result in results] == [1, 2]


def test_rrf_limits_results() -> None:
    results = reciprocal_rank_fusion(
        {
            "text": [1, 2, 3],
            "visual": [3, 2, 1],
        },
        top_k=2,
    )

    assert len(results) == 2


def test_rrf_accepts_empty_source_rankings() -> None:
    results = reciprocal_rank_fusion(
        {
            "text": [],
            "visual": [2, 1],
        }
    )

    assert [result.page_number for result in results] == [2, 1]


@pytest.mark.parametrize(
    ("rankings", "message"),
    [
        ({}, "at least one ranking source is required"),
        ({"": [1]}, "source names must not be blank"),
        ({"text": [0]}, "page numbers must be positive"),
    ],
)
def test_rrf_rejects_invalid_rankings(
    rankings: dict[str, list[int]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        reciprocal_rank_fusion(rankings)


def test_rrf_rejects_negative_rank_constant() -> None:
    with pytest.raises(
        ValueError,
        match="rank_constant must not be negative",
    ):
        reciprocal_rank_fusion(
            {"text": [1]},
            rank_constant=-1,
        )


def test_rrf_rejects_non_positive_top_k() -> None:
    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        reciprocal_rank_fusion(
            {"text": [1]},
            top_k=0,
        )


def test_rrf_rejects_mismatched_weights() -> None:
    with pytest.raises(
        ValueError,
        match="weights must match ranking sources",
    ):
        reciprocal_rank_fusion(
            {
                "text": [1],
                "visual": [2],
            },
            weights={"text": 1.0},
        )


def test_rrf_rejects_non_positive_weight() -> None:
    with pytest.raises(
        ValueError,
        match="weights must be greater than zero",
    ):
        reciprocal_rank_fusion(
            {"text": [1]},
            weights={"text": 0.0},
        )
