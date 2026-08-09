"""OpenAI vision-language model implementation."""

from base64 import b64encode
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, cast

from openai import OpenAI
from pydantic import BaseModel, Field

from sightcite.answering.models import (
    AnswerDraft,
    EvidencePage,
)

DEFAULT_OPENAI_VLM_MODEL = "gpt-5.6-luna"

_SYSTEM_PROMPT = """\
Answer the question using only the supplied scientific-paper page images.

Each image is explicitly labeled with its PDF page number. Return the page
numbers that directly support the answer. If the evidence does not contain
enough information, abstain and return no cited pages. Never cite a page that
was not supplied.
"""

_IMAGE_MEDIA_TYPES = {
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


class _StructuredAnswer(BaseModel):
    """Structured response schema sent to OpenAI."""

    text: str = Field(
        description="Answer grounded only in the supplied evidence.",
    )
    cited_pages: list[int] = Field(
        description="PDF page numbers directly supporting the answer.",
    )
    abstained: bool = Field(
        description=("True when the supplied evidence is insufficient to answer."),
    )


class _ParsedResponse(Protocol):
    @property
    def output_parsed(self) -> object | None:
        """Return the parsed structured output."""
        ...


class _ResponsesApi(Protocol):
    def parse(
        self,
        *,
        model: str,
        input: list[dict[str, object]],
        text_format: type[BaseModel],
    ) -> _ParsedResponse:
        """Create and parse a structured response."""
        ...


class _OpenAIClient(Protocol):
    @property
    def responses(self) -> _ResponsesApi:
        """Return the Responses API client."""
        ...


class OpenAIVisionLanguageModel:
    """Generate cited answers using OpenAI image understanding."""

    def __init__(
        self,
        model: str = DEFAULT_OPENAI_VLM_MODEL,
        *,
        api_key: str | None = None,
        image_detail: str = "high",
        client: _OpenAIClient | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be blank")

        if image_detail not in {"auto", "low", "high"}:
            raise ValueError("image_detail must be auto, low, or high")

        if client is None:
            loaded_client = OpenAI(api_key=api_key)
            client = cast(_OpenAIClient, loaded_client)

        self._model = model
        self._image_detail = image_detail
        self._client = client

    def generate_answer(
        self,
        question: str,
        evidence: Sequence[EvidencePage],
    ) -> AnswerDraft:
        """Generate a structured answer from supplied page images."""
        if not question.strip():
            raise ValueError("question must not be blank")

        if not evidence:
            raise ValueError("at least one evidence page is required")

        content: list[dict[str, object]] = [
            {
                "type": "input_text",
                "text": f"Question: {question}",
            }
        ]

        for page in evidence:
            content.extend(
                [
                    {
                        "type": "input_text",
                        "text": (f"The next image is PDF page {page.page_number}."),
                    },
                    {
                        "type": "input_image",
                        "image_url": _image_data_url(page.image_path),
                        "detail": self._image_detail,
                    },
                ]
            )

        response = self._client.responses.parse(
            model=self._model,
            input=[
                {
                    "role": "system",
                    "content": _SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": content,
                },
            ],
            text_format=_StructuredAnswer,
        )
        parsed = response.output_parsed

        if not isinstance(parsed, _StructuredAnswer):
            raise RuntimeError("OpenAI response did not contain a parsed answer")

        return AnswerDraft(
            text=parsed.text,
            cited_pages=tuple(parsed.cited_pages),
            abstained=parsed.abstained,
        )


def _image_data_url(image_path: Path) -> str:
    media_type = _IMAGE_MEDIA_TYPES.get(image_path.suffix.lower())

    if media_type is None:
        raise ValueError(f"unsupported evidence image type: {image_path.suffix}")

    encoded = b64encode(image_path.read_bytes()).decode("ascii")

    return f"data:{media_type};base64,{encoded}"
