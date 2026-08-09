from collections.abc import Sequence
from pathlib import Path

import pytest

from sightcite.answering import (
    AnswerDraft,
    EvidencePage,
    GroundedAnswerService,
)
from sightcite.ingestion import RenderedPage
from sightcite.retrieval import FusedSearchResult


class FakeVisionLanguageModel:
    def __init__(self, draft: AnswerDraft) -> None:
        self.draft = draft
        self.calls: list[tuple[str, tuple[EvidencePage, ...]]] = []

    def generate_answer(
        self,
        question: str,
        evidence: Sequence[EvidencePage],
    ) -> AnswerDraft:
        self.calls.append(
            (
                question,
                tuple(evidence),
            )
        )
        return self.draft


def make_results(
    tmp_path: Path,
) -> list[FusedSearchResult]:
    results: list[FusedSearchResult] = []

    for page_number, score in (
        (1, 0.8),
        (2, 0.6),
    ):
        image_path = tmp_path / f"page_{page_number:04d}.png"
        image_path.touch()

        results.append(
            FusedSearchResult(
                rank=page_number,
                score=score,
                page=RenderedPage(
                    page_number=page_number,
                    image_path=image_path,
                    width=600,
                    height=800,
                ),
                source_ranks=(
                    ("text", page_number),
                    ("visual", page_number),
                ),
            )
        )

    return results


def test_service_generates_grounded_answer(
    tmp_path: Path,
) -> None:
    model = FakeVisionLanguageModel(
        AnswerDraft(
            text="The reported value is 24.1 percent.",
            cited_pages=(2, 1),
        )
    )
    service = GroundedAnswerService(model)

    answer = service.answer(
        "What value was reported?",
        make_results(tmp_path),
    )

    assert answer.question == "What value was reported?"
    assert answer.text == ("The reported value is 24.1 percent.")
    assert [citation.page_number for citation in answer.citations] == [2, 1]
    assert not answer.abstained

    question, evidence = model.calls[0]

    assert question == "What value was reported?"
    assert [page.page_number for page in evidence] == [1, 2]
    assert evidence[0].retrieval_rank == 1
    assert evidence[0].retrieval_score == 0.8
    assert evidence[0].source_ranks == (
        ("text", 1),
        ("visual", 1),
    )


def test_service_preserves_abstention(
    tmp_path: Path,
) -> None:
    model = FakeVisionLanguageModel(
        AnswerDraft(
            text="The evidence is insufficient.",
            cited_pages=(),
            abstained=True,
        )
    )
    service = GroundedAnswerService(model)

    answer = service.answer(
        "What is the missing value?",
        make_results(tmp_path),
    )

    assert answer.abstained
    assert answer.citations == ()


def test_service_rejects_citation_outside_evidence(
    tmp_path: Path,
) -> None:
    model = FakeVisionLanguageModel(
        AnswerDraft(
            text="Unsupported answer.",
            cited_pages=(3,),
        )
    )
    service = GroundedAnswerService(model)

    with pytest.raises(
        ValueError,
        match="outside supplied evidence: 3",
    ):
        service.answer(
            "What happened?",
            make_results(tmp_path),
        )


def test_service_rejects_blank_question(
    tmp_path: Path,
) -> None:
    model = FakeVisionLanguageModel(
        AnswerDraft(
            text="Answer",
            cited_pages=(1,),
        )
    )
    service = GroundedAnswerService(model)

    with pytest.raises(
        ValueError,
        match="question must not be blank",
    ):
        service.answer(" ", make_results(tmp_path))

    assert model.calls == []


def test_service_rejects_empty_retrieval_results() -> None:
    model = FakeVisionLanguageModel(
        AnswerDraft(
            text="Answer",
            cited_pages=(1,),
        )
    )
    service = GroundedAnswerService(model)

    with pytest.raises(
        ValueError,
        match="at least one retrieval result is required",
    ):
        service.answer("Question", [])

    assert model.calls == []


def test_service_rejects_duplicate_retrieval_pages(
    tmp_path: Path,
) -> None:
    model = FakeVisionLanguageModel(
        AnswerDraft(
            text="Answer",
            cited_pages=(1,),
        )
    )
    service = GroundedAnswerService(model)
    results = make_results(tmp_path)
    duplicate = FusedSearchResult(
        rank=3,
        score=0.4,
        page=results[0].page,
        source_ranks=(
            ("text", 3),
            ("visual", 2),
        ),
    )

    with pytest.raises(
        ValueError,
        match="page numbers must be unique",
    ):
        service.answer(
            "Question",
            [*results, duplicate],
        )

    assert model.calls == []
