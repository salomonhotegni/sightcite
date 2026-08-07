import os
from pathlib import Path

import pytest

from sightcite.pipelines import FusedRetrievalPipeline
from sightcite.retrieval import (
    BgeTextEmbedder,
    ClipVisualEmbedder,
)

pytestmark = pytest.mark.model


@pytest.mark.skipif(
    os.environ.get("SIGHTCITE_RUN_MODEL_TESTS") != "1",
    reason="Set SIGHTCITE_RUN_MODEL_TESTS=1 to run model tests",
)
def test_fused_pipeline_with_real_models(
    sample_pdf: Path,
) -> None:
    text_embedder = BgeTextEmbedder(
        device="cpu",
        batch_size=2,
    )
    visual_embedder = ClipVisualEmbedder(
        device="cpu",
        batch_size=2,
    )

    with FusedRetrievalPipeline(
        sample_pdf,
        text_embedder,
        visual_embedder,
        visual_dpi=100,
    ) as pipeline:
        results = pipeline.search(
            "What is on the first page?",
            top_k=2,
        )
        image_paths = [result.page.image_path for result in results]

        assert pipeline.page_count == 2
        assert len(results) == 2
        assert {result.page.page_number for result in results} == {1, 2}
        assert [result.rank for result in results] == [1, 2]
        assert all(
            {source for source, _ in result.source_ranks} == {"text", "visual"}
            for result in results
        )
        assert all(path.is_file() for path in image_paths)

    assert all(not path.exists() for path in image_paths)
