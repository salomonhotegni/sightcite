"""Grounding enforcement for vision-language answers."""

from collections.abc import Sequence

from sightcite.answering.models import (
    EvidencePage,
    GroundedAnswer,
)
from sightcite.answering.vlm import VisionLanguageModel
from sightcite.retrieval import FusedSearchResult


class GroundedAnswerService:
    """Generate answers whose citations resolve to supplied evidence."""

    def __init__(
        self,
        model: VisionLanguageModel,
    ) -> None:
        self._model = model

    def answer(
        self,
        question: str,
        retrieval_results: Sequence[FusedSearchResult],
    ) -> GroundedAnswer:
        """Generate and validate an evidence-grounded answer."""
        if not question.strip():
            raise ValueError("question must not be blank")

        if not retrieval_results:
            raise ValueError("at least one retrieval result is required")

        evidence = tuple(
            EvidencePage(
                page_number=result.page.page_number,
                image_path=result.page.image_path,
                retrieval_rank=result.rank,
                retrieval_score=result.score,
                source_ranks=result.source_ranks,
            )
            for result in retrieval_results
        )
        evidence_by_page = {page.page_number: page for page in evidence}

        if len(evidence_by_page) != len(evidence):
            raise ValueError("retrieval result page numbers must be unique")

        draft = self._model.generate_answer(
            question,
            evidence,
        )

        if draft.abstained:
            return GroundedAnswer(
                question=question,
                text=draft.text,
                citations=(),
                abstained=True,
            )

        unknown_pages = set(draft.cited_pages) - set(evidence_by_page)

        if unknown_pages:
            formatted_pages = ", ".join(str(page_number) for page_number in sorted(unknown_pages))
            raise ValueError(f"model cited pages outside supplied evidence: {formatted_pages}")

        citations = tuple(evidence_by_page[page_number] for page_number in draft.cited_pages)

        return GroundedAnswer(
            question=question,
            text=draft.text,
            citations=citations,
        )
