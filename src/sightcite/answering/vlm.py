"""Vision-language model interface."""

from collections.abc import Sequence
from typing import Protocol

from sightcite.answering.models import (
    AnswerDraft,
    EvidencePage,
)


class VisionLanguageModel(Protocol):
    """Generate a structured answer from retrieved page evidence."""

    def generate_answer(
        self,
        question: str,
        evidence: Sequence[EvidencePage],
    ) -> AnswerDraft:
        """Answer a question using only the supplied evidence."""
        ...
