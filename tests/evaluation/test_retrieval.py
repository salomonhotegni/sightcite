import pytest

from sightcite.evaluation import RetrievalExample, evaluate_retrieval


def test_evaluate_retrieval_computes_page_metrics() -> None:
    examples = [
        RetrievalExample(
            query_id="q1",
            query="first question",
            relevant_pages=frozenset({2}),
        ),
        RetrievalExample(
            query_id="q2",
            query="second question",
            relevant_pages=frozenset({1, 3}),
        ),
    ]

    rankings = {
        "first question": [1, 1, 2, 3],
        "second question": [3, 2, 1],
    }

    evaluation = evaluate_retrieval(
        examples,
        lambda query: rankings[query],
    )

    assert evaluation.metrics.query_count == 2
    assert evaluation.metrics.recall_at_1 == pytest.approx(0.25)
    assert evaluation.metrics.recall_at_3 == pytest.approx(1.0)
    assert evaluation.metrics.recall_at_5 == pytest.approx(1.0)
    assert evaluation.metrics.mean_reciprocal_rank == pytest.approx(0.75)

    first_result = evaluation.queries[0]

    assert first_result.query_id == "q1"
    assert first_result.query == "first question"
    assert first_result.relevant_pages == frozenset({2})
    assert first_result.retrieved_pages == (1, 2, 3)
    assert first_result.recall_at_1 == 0.0
    assert first_result.recall_at_3 == 1.0
    assert first_result.recall_at_5 == 1.0
    assert first_result.reciprocal_rank == pytest.approx(0.5)


def test_evaluate_retrieval_assigns_zero_when_nothing_is_relevant() -> None:
    example = RetrievalExample(
        query_id="q1",
        query="question",
        relevant_pages=frozenset({4}),
    )

    evaluation = evaluate_retrieval(
        [example],
        lambda query: [1, 2, 3],
    )

    result = evaluation.queries[0]

    assert result.recall_at_1 == 0.0
    assert result.recall_at_3 == 0.0
    assert result.recall_at_5 == 0.0
    assert result.reciprocal_rank == 0.0


def test_evaluate_retrieval_accepts_empty_ranking() -> None:
    example = RetrievalExample(
        query_id="q1",
        query="question",
        relevant_pages=frozenset({1}),
    )

    evaluation = evaluate_retrieval([example], lambda query: [])

    assert evaluation.queries[0].retrieved_pages == ()


def test_evaluate_retrieval_rejects_empty_examples() -> None:
    with pytest.raises(
        ValueError,
        match="at least one retrieval example is required",
    ):
        evaluate_retrieval([], lambda query: [])


def test_evaluate_retrieval_rejects_invalid_retrieved_page() -> None:
    example = RetrievalExample(
        query_id="q1",
        query="question",
        relevant_pages=frozenset({1}),
    )

    with pytest.raises(
        ValueError,
        match="retrieved page numbers must be positive",
    ):
        evaluate_retrieval([example], lambda query: [0])


@pytest.mark.parametrize(
    ("query_id", "query", "relevant_pages", "message"),
    [
        ("", "question", frozenset({1}), "query_id must not be blank"),
        ("q1", "", frozenset({1}), "query must not be blank"),
        ("q1", "question", frozenset(), "relevant_pages must not be empty"),
        (
            "q1",
            "question",
            frozenset({0}),
            "relevant page numbers must be positive",
        ),
    ],
)
def test_retrieval_example_rejects_invalid_values(
    query_id: str,
    query: str,
    relevant_pages: frozenset[int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        RetrievalExample(
            query_id=query_id,
            query=query,
            relevant_pages=relevant_pages,
        )
