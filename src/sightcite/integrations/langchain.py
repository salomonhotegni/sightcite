"""LangChain orchestration for grounded question answering."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from langchain_core.runnables import Runnable, RunnableLambda

from sightcite.answering import GroundedAnswer
from sightcite.retrieval import FusedSearchResult


class RetrievalPipeline(Protocol):
    """Retrieval dependency required by the answer workflow."""

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[FusedSearchResult]:
        """Return retrieval results for a query."""
        ...


class AnswerService(Protocol):
    """Answering dependency required by the workflow."""

    def answer(
        self,
        question: str,
        retrieval_results: Sequence[FusedSearchResult],
    ) -> GroundedAnswer:
        """Generate a grounded answer from retrieval results."""
        ...


@dataclass(frozen=True, slots=True)
class AnswerRequest:
    """Input accepted by the LangChain answering workflow."""

    question: str
    top_k: int = 3

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("question must not be blank")

        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero")


@dataclass(frozen=True, slots=True)
class _RetrievedAnswerRequest:
    """Question paired with its retrieved evidence."""

    question: str
    retrieval_results: tuple[FusedSearchResult, ...]


def create_grounded_answer_chain(
    retrieval_pipeline: RetrievalPipeline,
    answer_service: AnswerService,
) -> Runnable[AnswerRequest, GroundedAnswer]:
    """Compose retrieval and grounded answering as a LangChain runnable."""

    def retrieve(request: AnswerRequest) -> _RetrievedAnswerRequest:
        results = retrieval_pipeline.search(
            request.question,
            top_k=request.top_k,
        )

        return _RetrievedAnswerRequest(
            question=request.question,
            retrieval_results=tuple(results),
        )

    def answer(request: _RetrievedAnswerRequest) -> GroundedAnswer:
        return answer_service.answer(
            request.question,
            request.retrieval_results,
        )

    retrieval_step: Runnable[
        AnswerRequest,
        _RetrievedAnswerRequest,
    ] = RunnableLambda(
        retrieve,
        name="sightcite_retrieve",
    )
    answering_step: Runnable[
        _RetrievedAnswerRequest,
        GroundedAnswer,
    ] = RunnableLambda(
        answer,
        name="sightcite_answer",
    )

    return retrieval_step | answering_step
