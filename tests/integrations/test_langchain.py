from collections.abc import Sequence
from pathlib import Path

import pytest

from sightcite.answering import EvidencePage, GroundedAnswer
from sightcite.ingestion import RenderedPage
from sightcite.integrations.langchain import (
    AnswerRequest,
    create_grounded_answer_chain,
)
from sightcite.retrieval import FusedSearchResult


class FakeRetrievalPipeline:
    def __init__(
        self,
        results: list[FusedSearchResult],
    ) -> None:
        self._results = results
        self.calls: list[tuple[str, int]] = []

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[FusedSearchResult]:
        self.calls.append((query, top_k))
        return self._results[:top_k]


class FakeAnswerService:
    def __init__(
        self,
        answer: GroundedAnswer,
    ) -> None:
        self._answer = answer
        self.calls: list[
            tuple[
                str,
                tuple[FusedSearchResult, ...],
            ]
        ] = []

    def answer(
        self,
        question: str,
        retrieval_results: Sequence[FusedSearchResult],
    ) -> GroundedAnswer:
        self.calls.append(
            (
                question,
                tuple(retrieval_results),
            )
        )
        return self._answer


def make_retrieval_result(
    image_path: Path,
) -> FusedSearchResult:
    return FusedSearchResult(
        rank=1,
        score=0.75,
        page=RenderedPage(
            page_number=2,
            image_path=image_path,
            width=800,
            height=1200,
        ),
        source_ranks=(
            ("text", 1),
            ("visual", 2),
        ),
    )


def test_grounded_answer_chain_composes_retrieval_and_answering(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page-2.png"
    image_path.touch()

    retrieval_result = make_retrieval_result(image_path)
    expected_answer = GroundedAnswer(
        question="What did the experiment show?",
        text="The experiment showed improved performance.",
        citations=(
            EvidencePage(
                page_number=2,
                image_path=image_path,
                retrieval_rank=1,
                retrieval_score=0.75,
                source_ranks=(
                    ("text", 1),
                    ("visual", 2),
                ),
            ),
        ),
    )
    retrieval_pipeline = FakeRetrievalPipeline([retrieval_result])
    answer_service = FakeAnswerService(expected_answer)
    chain = create_grounded_answer_chain(
        retrieval_pipeline,
        answer_service,
    )

    result = chain.invoke(
        AnswerRequest(
            question="What did the experiment show?",
            top_k=4,
        )
    )

    assert result is expected_answer
    assert retrieval_pipeline.calls == [
        ("What did the experiment show?", 4),
    ]
    assert answer_service.calls == [
        (
            "What did the experiment show?",
            (retrieval_result,),
        ),
    ]


def test_answer_request_uses_default_top_k(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page-2.png"
    image_path.touch()

    retrieval_result = make_retrieval_result(image_path)
    expected_answer = GroundedAnswer(
        question="Question?",
        text="Answer.",
        citations=(
            EvidencePage(
                page_number=2,
                image_path=image_path,
                retrieval_rank=1,
                retrieval_score=0.75,
                source_ranks=(
                    ("text", 1),
                    ("visual", 2),
                ),
            ),
        ),
    )
    retrieval_pipeline = FakeRetrievalPipeline([retrieval_result])
    chain = create_grounded_answer_chain(
        retrieval_pipeline,
        FakeAnswerService(expected_answer),
    )

    chain.invoke(AnswerRequest(question="Question?"))

    assert retrieval_pipeline.calls == [("Question?", 3)]


@pytest.mark.parametrize(
    ("question", "top_k", "message"),
    [
        ("", 3, "question must not be blank"),
        ("   ", 3, "question must not be blank"),
        ("Question?", 0, "top_k must be greater than zero"),
        ("Question?", -1, "top_k must be greater than zero"),
    ],
)
def test_answer_request_rejects_invalid_values(
    question: str,
    top_k: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        AnswerRequest(
            question=question,
            top_k=top_k,
        )
