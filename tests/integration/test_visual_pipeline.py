import os
from pathlib import Path

import numpy as np
import pytest

from sightcite.pipelines import VisualRetrievalPipeline
from sightcite.retrieval import ClipVisualEmbedder

pytestmark = pytest.mark.model


@pytest.mark.skipif(
    os.environ.get("SIGHTCITE_RUN_MODEL_TESTS") != "1",
    reason="Set SIGHTCITE_RUN_MODEL_TESTS=1 to run model tests",
)
def test_visual_pipeline_with_real_clip_model(
    sample_pdf: Path,
) -> None:
    pdf_path = sample_pdf

    embedder = ClipVisualEmbedder(
        device="cpu",
        batch_size=2,
    )

    with VisualRetrievalPipeline(
        pdf_path,
        embedder,
        dpi=100,
    ) as pipeline:
        results = pipeline.search(
            "a page from a scientific paper",
            top_k=2,
        )
        image_paths = [result.page.image_path for result in results]

        assert pipeline.page_count == 2
        assert len(results) == 2
        assert {result.page.page_number for result in results} == {1, 2}
        assert [result.rank for result in results] == [1, 2]
        assert results[0].score >= results[1].score
        assert all(np.isfinite(result.score) for result in results)
        assert all(path.is_file() for path in image_paths)

    assert all(not path.exists() for path in image_paths)
