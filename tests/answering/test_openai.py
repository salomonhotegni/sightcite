from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
from pydantic import BaseModel

from sightcite.answering import (
    AnswerDraft,
    EvidencePage,
    OpenAIVisionLanguageModel,
)


@dataclass
class FakeParsedResponse:
    output_parsed: object | None


@dataclass
class RecordedCall:
    model: str
    input: list[dict[str, object]]
    text_format: type[BaseModel]


class FakeResponsesApi:
    def __init__(
        self,
        payload: dict[str, object] | None,
    ) -> None:
        self.payload = payload
        self.calls: list[RecordedCall] = []

    def parse(
        self,
        *,
        model: str,
        input: list[dict[str, object]],
        text_format: type[BaseModel],
    ) -> FakeParsedResponse:
        self.calls.append(
            RecordedCall(
                model=model,
                input=input,
                text_format=text_format,
            )
        )

        if self.payload is None:
            return FakeParsedResponse(None)

        return FakeParsedResponse(text_format.model_validate(self.payload))


class FakeOpenAIClient:
    def __init__(
        self,
        payload: dict[str, object] | None,
    ) -> None:
        self.responses = FakeResponsesApi(payload)


@pytest.fixture
def evidence_page(tmp_path: Path) -> EvidencePage:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"page image")

    return EvidencePage(
        page_number=2,
        image_path=image_path,
        retrieval_rank=1,
        retrieval_score=0.8,
    )


def test_openai_model_generates_structured_answer(
    evidence_page: EvidencePage,
) -> None:
    client = FakeOpenAIClient(
        {
            "text": "The reported value is 24.1 percent.",
            "cited_pages": [2],
            "abstained": False,
        }
    )
    model = OpenAIVisionLanguageModel(
        model="test-model",
        image_detail="high",
        client=client,
    )

    draft = model.generate_answer(
        "What value was reported?",
        [evidence_page],
    )

    assert draft == AnswerDraft(
        text="The reported value is 24.1 percent.",
        cited_pages=(2,),
    )

    call = client.responses.calls[0]

    assert call.model == "test-model"
    assert call.text_format.__name__ == "_StructuredAnswer"

    user_message = call.input[1]
    content = cast(
        list[dict[str, object]],
        user_message["content"],
    )

    assert content[0] == {
        "type": "input_text",
        "text": "Question: What value was reported?",
    }
    assert content[1]["text"] == ("The next image is PDF page 2.")
    assert content[2]["type"] == "input_image"
    assert content[2]["detail"] == "high"
    assert cast(str, content[2]["image_url"]).startswith("data:image/png;base64,")


def test_openai_model_preserves_abstention(
    evidence_page: EvidencePage,
) -> None:
    client = FakeOpenAIClient(
        {
            "text": "The supplied evidence is insufficient.",
            "cited_pages": [],
            "abstained": True,
        }
    )
    model = OpenAIVisionLanguageModel(client=client)

    draft = model.generate_answer(
        "What is the missing value?",
        [evidence_page],
    )

    assert draft.abstained
    assert draft.cited_pages == ()


def test_openai_model_rejects_missing_parsed_output(
    evidence_page: EvidencePage,
) -> None:
    model = OpenAIVisionLanguageModel(client=FakeOpenAIClient(None))

    with pytest.raises(
        RuntimeError,
        match="did not contain a parsed answer",
    ):
        model.generate_answer(
            "Question",
            [evidence_page],
        )


def test_openai_model_rejects_blank_question(
    evidence_page: EvidencePage,
) -> None:
    client = FakeOpenAIClient(None)
    model = OpenAIVisionLanguageModel(client=client)

    with pytest.raises(
        ValueError,
        match="question must not be blank",
    ):
        model.generate_answer(" ", [evidence_page])

    assert client.responses.calls == []


def test_openai_model_rejects_empty_evidence() -> None:
    client = FakeOpenAIClient(None)
    model = OpenAIVisionLanguageModel(client=client)

    with pytest.raises(
        ValueError,
        match="at least one evidence page is required",
    ):
        model.generate_answer("Question", [])

    assert client.responses.calls == []


@pytest.mark.parametrize(
    "image_detail",
    ["", "original", "invalid"],
)
def test_openai_model_rejects_invalid_image_detail(
    image_detail: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="image_detail must be auto, low, or high",
    ):
        OpenAIVisionLanguageModel(
            image_detail=image_detail,
            client=FakeOpenAIClient(None),
        )


def test_openai_model_rejects_blank_model() -> None:
    with pytest.raises(
        ValueError,
        match="model must not be blank",
    ):
        OpenAIVisionLanguageModel(
            model=" ",
            client=FakeOpenAIClient(None),
        )


def test_openai_model_rejects_unsupported_image_type(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page.bmp"
    image_path.write_bytes(b"image")
    evidence = EvidencePage(
        page_number=1,
        image_path=image_path,
        retrieval_rank=1,
        retrieval_score=1.0,
    )
    model = OpenAIVisionLanguageModel(client=FakeOpenAIClient(None))

    with pytest.raises(
        ValueError,
        match="unsupported evidence image type",
    ):
        model.generate_answer("Question", [evidence])
