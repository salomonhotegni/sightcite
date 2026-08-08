"""Models for evidence-grounded answering."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path


@dataclass(frozen=True, slots=True)
class EvidencePage:
    """One retrieved page supplied to an answering model."""

    page_number: int
    image_path: Path
    retrieval_rank: int
    retrieval_score: float
    source_ranks: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        if self.page_number <= 0:
            raise ValueError("evidence page number must be positive")

        if not self.image_path.is_file():
            raise FileNotFoundError(f"evidence image does not exist: {self.image_path}")

        if self.retrieval_rank <= 0:
            raise ValueError("evidence retrieval rank must be positive")

        if not isfinite(self.retrieval_score):
            raise ValueError("evidence retrieval score must be finite")

        source_names: set[str] = set()

        for source_name, source_rank in self.source_ranks:
            if not source_name.strip():
                raise ValueError("evidence source names must not be blank")

            if source_name in source_names:
                raise ValueError("evidence source names must be unique")

            if source_rank <= 0:
                raise ValueError("evidence source ranks must be positive")

            source_names.add(source_name)


@dataclass(frozen=True, slots=True)
class AnswerDraft:
    """Raw structured output produced by an answering model."""

    text: str
    cited_pages: tuple[int, ...]
    abstained: bool = False

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("answer text must not be blank")

        if any(page_number <= 0 for page_number in self.cited_pages):
            raise ValueError("cited page numbers must be positive")

        if len(self.cited_pages) != len(set(self.cited_pages)):
            raise ValueError("cited page numbers must be unique")

        if self.abstained and self.cited_pages:
            raise ValueError("an abstained answer must not contain citations")

        if not self.abstained and not self.cited_pages:
            raise ValueError("a non-abstained answer must contain citations")


@dataclass(frozen=True, slots=True)
class GroundedAnswer:
    """Validated answer linked to retrieved evidence pages."""

    question: str
    text: str
    citations: tuple[EvidencePage, ...]
    abstained: bool = False

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("question must not be blank")

        if not self.text.strip():
            raise ValueError("answer text must not be blank")

        citation_pages = [citation.page_number for citation in self.citations]

        if len(citation_pages) != len(set(citation_pages)):
            raise ValueError("answer citation pages must be unique")

        if self.abstained and self.citations:
            raise ValueError("an abstained answer must not contain citations")

        if not self.abstained and not self.citations:
            raise ValueError("a non-abstained answer must contain citations")
