import os
from pathlib import Path

import pytest

from sightcite.answering import (
    EvidencePage,
    OpenAIVisionLanguageModel,
)
from sightcite.ingestion import render_pdf

pytestmark = pytest.mark.openai


@pytest.mark.skipif(
    os.environ.get("SIGHTCITE_RUN_OPENAI_TESTS") != "1",
    reason="Set SIGHTCITE_RUN_OPENAI_TESTS=1 to run OpenAI tests",
)
def test_openai_vlm_answers_from_real_page_image(
    sample_pdf: Path,
    tmp_path: Path,
) -> None:
    rendered_pages = render_pdf(
        sample_pdf,
        tmp_path / "rendered",
        dpi=144,
    )
    first_page = rendered_pages[0]
    evidence = EvidencePage(
        page_number=first_page.page_number,
        image_path=first_page.image_path,
        retrieval_rank=1,
        retrieval_score=1.0,
    )

    model = OpenAIVisionLanguageModel()
    draft = model.generate_answer(
        "What words are shown on this page?",
        [evidence],
    )

    assert not draft.abstained
    assert draft.text.strip()
    assert draft.cited_pages == (1,)
