from pathlib import Path

import pytest

from sightcite.answering import (
    AnswerDraft,
    EvidencePage,
    GroundedAnswer,
)


@pytest.fixture
def evidence_page(tmp_path: Path) -> EvidencePage:
    image_path = tmp_path / "page.png"
    image_path.touch()

    return EvidencePage(
        page_number=2,
        image_path=image_path,
        retrieval_rank=1,
        retrieval_score=0.75,
        source_ranks=(
            ("text", 2),
            ("visual", 1),
        ),
    )


def test_evidence_page_preserves_retrieval_metadata(
    evidence_page: EvidencePage,
) -> None:
    assert evidence_page.page_number == 2
    assert evidence_page.retrieval_rank == 1
    assert evidence_page.retrieval_score == 0.75
    assert evidence_page.source_ranks == (
        ("text", 2),
        ("visual", 1),
    )


def test_evidence_page_rejects_missing_image(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        FileNotFoundError,
        match="evidence image does not exist",
    ):
        EvidencePage(
            page_number=1,
            image_path=tmp_path / "missing.png",
            retrieval_rank=1,
            retrieval_score=1.0,
        )


@pytest.mark.parametrize(
    ("page_number", "retrieval_rank", "score", "message"),
    [
        (0, 1, 1.0, "page number must be positive"),
        (1, 0, 1.0, "retrieval rank must be positive"),
        (1, 1, float("nan"), "score must be finite"),
    ],
)
def test_evidence_page_rejects_invalid_metadata(
    tmp_path: Path,
    page_number: int,
    retrieval_rank: int,
    score: float,
    message: str,
) -> None:
    image_path = tmp_path / "page.png"
    image_path.touch()

    with pytest.raises(ValueError, match=message):
        EvidencePage(
            page_number=page_number,
            image_path=image_path,
            retrieval_rank=retrieval_rank,
            retrieval_score=score,
        )


@pytest.mark.parametrize(
    ("source_ranks", "message"),
    [
        ((("", 1),), "source names must not be blank"),
        (
            (("text", 1), ("text", 2)),
            "source names must be unique",
        ),
        ((("text", 0),), "source ranks must be positive"),
    ],
)
def test_evidence_page_rejects_invalid_source_ranks(
    tmp_path: Path,
    source_ranks: tuple[tuple[str, int], ...],
    message: str,
) -> None:
    image_path = tmp_path / "page.png"
    image_path.touch()

    with pytest.raises(ValueError, match=message):
        EvidencePage(
            page_number=1,
            image_path=image_path,
            retrieval_rank=1,
            retrieval_score=1.0,
            source_ranks=source_ranks,
        )


def test_answer_draft_accepts_cited_answer() -> None:
    draft = AnswerDraft(
        text="The reported value is 24.1 percent.",
        cited_pages=(1,),
    )

    assert not draft.abstained


def test_answer_draft_accepts_abstention() -> None:
    draft = AnswerDraft(
        text="The supplied evidence does not answer the question.",
        cited_pages=(),
        abstained=True,
    )

    assert draft.abstained


@pytest.mark.parametrize(
    ("draft", "message"),
    [
        (
            {
                "text": " ",
                "cited_pages": (1,),
            },
            "answer text must not be blank",
        ),
        (
            {
                "text": "Answer",
                "cited_pages": (),
            },
            "must contain citations",
        ),
        (
            {
                "text": "Answer",
                "cited_pages": (1, 1),
            },
            "must be unique",
        ),
        (
            {
                "text": "No answer",
                "cited_pages": (1,),
                "abstained": True,
            },
            "must not contain citations",
        ),
    ],
)
def test_answer_draft_rejects_invalid_output(
    draft: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        AnswerDraft(**draft)  # type: ignore[arg-type]


def test_grounded_answer_links_citations(
    evidence_page: EvidencePage,
) -> None:
    answer = GroundedAnswer(
        question="What value was reported?",
        text="The value was 24.1 percent.",
        citations=(evidence_page,),
    )

    assert answer.citations[0].page_number == 2


def test_answer_draft_rejects_invalid_cited_page() -> None:
    with pytest.raises(
        ValueError,
        match="cited page numbers must be positive",
    ):
        AnswerDraft(
            text="Answer",
            cited_pages=(0,),
        )


def test_grounded_answer_accepts_abstention() -> None:
    answer = GroundedAnswer(
        question="What value was reported?",
        text="The supplied evidence does not answer the question.",
        citations=(),
        abstained=True,
    )

    assert answer.abstained


def test_grounded_answer_rejects_blank_question(
    evidence_page: EvidencePage,
) -> None:
    with pytest.raises(
        ValueError,
        match="question must not be blank",
    ):
        GroundedAnswer(
            question=" ",
            text="Answer",
            citations=(evidence_page,),
        )


def test_grounded_answer_rejects_blank_text(
    evidence_page: EvidencePage,
) -> None:
    with pytest.raises(
        ValueError,
        match="answer text must not be blank",
    ):
        GroundedAnswer(
            question="Question",
            text=" ",
            citations=(evidence_page,),
        )


def test_grounded_answer_rejects_duplicate_citations(
    evidence_page: EvidencePage,
) -> None:
    with pytest.raises(
        ValueError,
        match="citation pages must be unique",
    ):
        GroundedAnswer(
            question="Question",
            text="Answer",
            citations=(
                evidence_page,
                evidence_page,
            ),
        )


def test_grounded_answer_rejects_cited_abstention(
    evidence_page: EvidencePage,
) -> None:
    with pytest.raises(
        ValueError,
        match="must not contain citations",
    ):
        GroundedAnswer(
            question="Question",
            text="No answer",
            citations=(evidence_page,),
            abstained=True,
        )


def test_grounded_answer_requires_citation() -> None:
    with pytest.raises(
        ValueError,
        match="must contain citations",
    ):
        GroundedAnswer(
            question="Question",
            text="Answer",
            citations=(),
        )
