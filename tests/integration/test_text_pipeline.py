import os
from pathlib import Path

import pytest

from sightcite.pipelines import TextRetrievalPipeline
from sightcite.retrieval import BgeTextEmbedder

RUN_MODEL_TESTS = os.getenv("SIGHTCITE_RUN_MODEL_TESTS") == "1"


@pytest.mark.model
@pytest.mark.skipif(
    not RUN_MODEL_TESTS,
    reason="set SIGHTCITE_RUN_MODEL_TESTS=1 to run model tests",
)
def test_text_pipeline_with_real_bge_model(sample_pdf: Path) -> None:
    embedder = BgeTextEmbedder(device="cpu")
    pipeline = TextRetrievalPipeline(sample_pdf, embedder)

    results = pipeline.search("What is on the second page?", top_k=1)

    assert len(results) == 1
    assert results[0].chunk.page_number in {1, 2}
    assert -1.0 <= results[0].score <= 1.0
